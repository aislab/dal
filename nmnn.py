import numpy as np
import jax
from jax import numpy as jp
import config as cfg
import utils
from dataclasses import dataclass, field
from typing import Union
from jax.tree_util import tree_structure
from functools import partial
import optax


# convenience global for NN inditification in prints
individual_id = None


# print with process identification
def pprint(*m):
    if not cfg.suppress_nn_prints:
        print('[NN:'+str(individual_id)+']', *m)


# make random projection weights.
def make_projection_weights(rng_key,ni,nj):
    r = jp.sqrt(1/ni)
    w = jax.random.uniform(rng_key,(ni,nj),minval=-r,maxval=r)
    return w
    

# make an activatory projection.
def make_projection_a(rng_key,ni,nj,origin):
    w = make_projection_weights(rng_key,ni,nj)
    return Projection(w,origin,origin)


# make a modulatory projection.
def make_projection_m(rng_key,nm,ni,nj,origin,for_guided_init):

    rng_key, *kk = jax.random.split(rng_key,7)
    priority = jax.random.uniform(kk[0])
    beta_multiplier = 0.01
    tgtw_multiplier = 0.01 if for_guided_init else 1.0
    eta_multiplier = 0.01

    hidden_multiplier = 0.01 if for_guided_init else 1.0
    wih = hidden_multiplier*make_projection_weights(kk[1],nm+1,cfg.nm_fm_hidden_column_size)
    who_beta = beta_multiplier*make_projection_weights(kk[2],cfg.nm_fm_hidden_column_size,ni*nj)
    who_tgtw = tgtw_multiplier*make_projection_weights(kk[3],cfg.nm_fm_hidden_column_size,ni*nj)
    ww = [wih,who_beta,who_tgtw]

    if cfg.nm_eta_enabled:
        input_dim = 1
        if cfg.eta_sees_k: input_dim+=nm+1
        if cfg.eta_sees_i: input_dim+=ni
        if cfg.eta_sees_j: input_dim+=nj
        wih_eta = eta_multiplier*make_projection_weights(kk[4],input_dim,cfg.nm_fg_hidden_column_size)
        eta_res = 1 if cfg.projection_level_eta else ni*nj
        who_eta = eta_multiplier*make_projection_weights(kk[5],cfg.nm_fg_hidden_column_size,eta_res)
        ww = ww+[wih_eta,who_eta]

    return Projection(ww,origin,origin,priority)
        

# check if there are paths from the input column to all output columns.
def check_connected(grid):
    if not cfg.enforce_connectedness:
        return True
    n_columns = grid.shape[0]
    for j in range(n_columns-cfg.n_output_columns,n_columns):
        connected = False
        check_set = list(np.where(grid[:,j])[0])
        for i_column in check_set:
            if i_column < cfg.n_input_columns:
                connected = True
                break
            check_set += list(np.where(grid[:i_column,i_column])[0])
        if not connected: return False
    return True
        

# forward propagation.
@partial(jax.jit,static_argnums=(2,))
def dense_forward(w,a,activation_function):
    # set bias neuron activation to 1
    a = a.at[...,-1].set(1)
    if activation_function is None:
        a = jp.matmul(a,w)
    else:
        a = jp.matmul(activation_function(a),w)
    return a
    

# vmap map meep meep.
multi_w_dense_forward = jax.vmap(dense_forward,in_axes=[0,0,None])
multi_multi_w_dense_forward = jax.vmap(multi_w_dense_forward,in_axes=[0,0,None])
fixed_weight_multi_w_dense_forward = jax.vmap(dense_forward,in_axes=[None,0,None])
fixed_weight_multi_multi_w_dense_forward = jax.vmap(fixed_weight_multi_w_dense_forward,in_axes=[None,0,None])


# gradient clipping.
def clip_grad_norm(grad, max_norm):
    norm = jp.linalg.norm(jp.array(jax.tree_util.tree_leaves(jax.tree.map(jp.linalg.norm,grad))))
    clip = lambda x: jp.where(norm<max_norm,x,x*max_norm/(norm+1e-6))
    return jax.tree_util.tree_map(clip, grad)
    

# calculate beta and target weight values for modulatory projection.
@jax.jit
def nm_beta_and_tgtw(a,nm_proj_w):
    wih,whr,wht = nm_proj_w[:3]
    ah = fixed_weight_multi_w_dense_forward(wih,a,cfg.activation_function_hidden_nm)
    beta = cfg.nm_update_activation_function_r(fixed_weight_multi_w_dense_forward(whr,ah,cfg.activation_function_hidden_nm))
    tgtw = cfg.nm_update_activation_function_t(fixed_weight_multi_w_dense_forward(wht,ah,cfg.activation_function_hidden_nm))
    return beta, tgtw

multi_nm_beta_and_tgtw = jax.vmap(nm_beta_and_tgtw,in_axes=[0,None])


# calculate eta values for modulatory projection.
@jax.jit
def nm_eta(am,ai,w,nm_proj_w):
    wih, who = nm_proj_w[-2:]
    aj_local = multi_w_dense_forward(w,ai,cfg.activation_function_hidden)
    bias_shape = ai.shape[:-1]+(1,)
    bias = jp.ones(bias_shape)
    a = [bias]
    if cfg.eta_sees_k: a.append(am)
    if cfg.eta_sees_i: a.append(ai)
    if cfg.eta_sees_j: a.append(aj_local)
    a = jp.concatenate(a,axis=-1)
    ah = fixed_weight_multi_w_dense_forward(wih,a,cfg.activation_function_hidden_nm)
    eta = cfg.nm_update_activation_function_g(fixed_weight_multi_w_dense_forward(who,ah,cfg.activation_function_hidden_nm))
    return eta[...,None]


# given target weights, beta, and activation of the pre-synaptic and modulatory columns, calculate the new weight.
# during guided initialisation, calculation of weight target and beta can be parallelised, but eta has to be calculated serially,
# because it depends on the activation of the post-synaptic neuron, which in turn depends on preceding weight updates.
@jax.jit
def w_update_nm(w,w_geno,beta,tgtw,am,ai,nm_proj_w):
    tgtw = tgtw.reshape(w.shape)
    if cfg.nm_target_relative_to_geno_w:
        tgtw += w_geno
    dw = tgtw-w
    if cfg.nm_beta_enabled:
        dw *= beta.reshape(w.shape)
    if cfg.nm_eta_enabled:
        eta = nm_eta(am,ai,w,nm_proj_w)
        if not cfg.projection_level_eta:
            eta = eta.reshape(w.shape)
        return w+eta*dw, eta
    else:
        return w+dw, 0


# given activation of the pre-synaptic, post-synaptic, and modulatory columns, perform the weight update for a modulatory projection.
@jax.jit
def nm_weight_update(am,ai,aj,w,nm_proj_w,w_geno):
    am = jp.pad(am,((0,0),(0,1)),mode='constant')
    beta, tgtw = nm_beta_and_tgtw(am,nm_proj_w)
    w, g = w_update_nm(w,w_geno,beta,tgtw,am,ai,nm_proj_w)
    return w, g

    
# loss function for guided initialisation of modulatory projections.
@partial(jax.jit,static_argnums=(0,8,9))
def guided_weight_init_loss(n_a,
                            activation_history_k,
                            activation_history_i,
                            activation_history_j,
                            w_geno,
                            w_final,
                            nm_proj_w,
                            fitness_weights,
                            activation_function_i,
                            activation_function_j):

    n_steps = len(activation_history_k)
    #t = jp.arange(n_steps)/n_steps
    #t = jp.repeat(t[:,None,None],cfg.n_task_instances,axis=1)
    #activation_history_k = jp.concatenate((activation_history_k,t),axis=2)
    activation_history_k = jp.pad(activation_history_k,((0,0),(0,0),(0,1)))
    w = jp.broadcast_to(w_geno[None],(cfg.n_task_instances,)+w_geno.shape)

    rate_seq, target_seq = multi_nm_beta_and_tgtw(activation_history_k,nm_proj_w)
    
    pprint('rate_seq:', rate_seq.shape, 'target_seq:', target_seq.shape, 'w:', w.shape, 'w_final:', w_final.shape)
    def f(carry,x):
        
        loss, w = carry
        beta, tgtw, am, ai, aj_postsum = x
        
        w_new, eta = w_update_nm(w,w_geno,beta,tgtw,am,ai,nm_proj_w)
        
        if cfg.guided_weight_modification_loss_locus == 'weight':
            if cfg.guided_weight_modification_diff_function == 'mae':
                l = ((jp.abs(w-w_final))*fitness_weights[:,None,None]).mean((1,2))
            if cfg.guided_weight_modification_diff_function == 'mse':
                l = (((w-w_final)**2)*fitness_weights[:,None,None]).mean((1,2))
        
        if cfg.guided_weight_modification_loss_locus == 'activation':
            
            aj_new = multi_w_dense_forward(w_new,ai,activation_function_i)
            aj_via_final = multi_w_dense_forward(w_final,ai,activation_function_i)
            
            if cfg.actvation_locus_loss_considers_activation_from_elsewhere:
                aj_local = multi_w_dense_forward(w_new,ai,activation_function_i)
                aj_from_elsewhere = aj_postsum-aj_local
                aj_new += aj_from_elsewhere
                aj_via_final += aj_from_elsewhere
            
            if cfg.actvation_locus_loss_applies_activation_functions:
                aj_new = activation_function_j(aj_new)
                aj_via_final = activation_function_j(aj_via_final)
                
            if cfg.guided_weight_modification_diff_function == 'mae':
                l = ((jp.abs(aj_new-aj_via_final))*fitness_weights[:,None]).mean(1)
            if cfg.guided_weight_modification_diff_function == 'mse':
                l = (((aj_new-aj_via_final)**2)*fitness_weights[:,None]).mean(1)
        
        if cfg.nm_beta_enabled and cfg.nm_beta_loss_weight>0:
            l += cfg.nm_beta_loss_weight*beta.mean(1)/n_steps
        if cfg.nm_eta_enabled and cfg.nm_eta_loss_weight>0:
            l += cfg.nm_eta_loss_weight*eta.mean((1,2))/n_steps
        
        return (loss+l,w_new), l
        
    loss = jp.zeros(cfg.n_task_instances)
    (loss,w),loss_seq = jax.lax.scan(f,(loss,w),(rate_seq,target_seq,activation_history_k,activation_history_i,activation_history_j))
    
    loss_train = loss[cfg.n_validation:].mean()
    loss_validation = loss[:cfg.n_validation].mean() if cfg.n_validation else 0
    return loss_train, (loss_validation, w)
    
    
guided_init_loss_grad = jax.value_and_grad(guided_weight_init_loss,argnums=(4,6),has_aux=True)


# attempt at calculating grads per step to reduce computation cost
@partial(jax.jit,static_argnums=(0,8,9))
def mGWM_sub_update(n_a,
                            activation_history_k,
                            activation_history_i,
                            activation_history_j,
                            w_geno,
                            w_final,
                            nm_proj_w,
                            fitness_weights,
                            activation_function_i,
                            activation_function_j,
                            lr):

    n_steps = len(activation_history_k)

    w = jp.broadcast_to(w_geno[None],w_final.shape)

    def f(nm_proj_w, i_step, w, ak, ai, aj_postsum):

        rates, targets = nm_beta_and_tgtw(ak,nm_proj_w)

        w = jax.lax.stop_gradient(w) # TODO: does this make a difference? (in performance and/or GPU memory use?) -> it should not, cause grads are calculated over f step by step
        #w, g = w_update_nm(w,rates,targets,ak,ai,nm_proj_w,w_geno)
        w, g = w_update_nm(w,w_geno,rates,targets,ak,ai,nm_proj_w)

        if cfg.guided_weight_modification_loss_locus == 'weight':
            if cfg.guided_weight_modification_diff_function == 'mae':
                l = ((jp.abs(w-w_final))*fitness_weights[:,None,None]).mean((1,2))
            if cfg.guided_weight_modification_diff_function == 'mse':
                l = (((w-w_final)**2)*fitness_weights[:,None,None]).mean((1,2))
            if cfg.guided_weight_modification_diff_function == 'mre':
                l = ((jp.abs(w-w_final)**.5)*fitness_weights[:,None,None]).mean((1,2))

        if cfg.guided_weight_modification_loss_locus == 'activation':

            aj_new = multi_w_dense_forward(w,ai,activation_function_i,True)
            aj_via_final = multi_w_dense_forward(w_final,ai,activation_function_i,True)

            if cfg.actvation_locus_loss_considers_activation_from_elsewhere:
                aj_local = multi_w_dense_forward(w,ai,activation_function_i,True)
                aj_from_elsewhere = aj_postsum-aj_local
                aj_new += aj_from_elsewhere
                aj_via_final += aj_from_elsewhere

            if cfg.actvation_locus_loss_applies_activation_functions:
                aj_new = activation_function_j(aj_new)
                aj_via_final = activation_function_j(aj_via_final)

            if cfg.guided_weight_modification_diff_function == 'mae':
                l = ((jp.abs(aj_new-aj_via_final))*fitness_weights[:,None]).mean(1)
            if cfg.guided_weight_modification_diff_function == 'mse':
                l = (((aj_new-aj_via_final)**2)*fitness_weights[:,None]).mean(1)
            if cfg.guided_weight_modification_diff_function == 'mre':
                l = ((jp.abs(aj_new-aj_via_final)**.5)*fitness_weights[:,None]).mean(1)

        if cfg.nm_beta_enabled and cfg.nm_beta_loss_weight>0:
            l += cfg.nm_beta_loss_weight*rates.mean(1)/n_steps
        if cfg.nm_eta_enabled and cfg.nm_eta_loss_weight>0:
            l += cfg.nm_eta_loss_weight*g.mean((1,2))/n_steps

        return l.sum(),(i_step+1,w,l)

    def f_with_grad(carry,x):
        loss, i_step, nm_proj_w, w, grads = carry
        ak, ai, aj_postsum = x
        (step_loss_scalar,(i_step,w,step_loss)),step_grads = jax.value_and_grad(f,argnums=0,has_aux=True)(nm_proj_w, i_step, w, ak, ai, aj_postsum)
        loss += step_loss

        if cfg.cheap_GWM_use_maxgrad:
            grads = [jp.where(jp.abs(sg)>=jp.abs(g),sg,g) for (g,sg) in zip(grads,step_grads)]
        else:
            grads = [(g+sg) for (g,sg) in zip(grads,step_grads)]

        return (loss, i_step, nm_proj_w, w, grads), None

    activation_history_k = jp.pad(activation_history_k,((0,0),(0,0),(0,1)))
    loss = jp.zeros(w_final.shape[0])
    grads = [jp.zeros_like(z) for z in nm_proj_w]
    (loss,_,_,w_new,grads),_ = jax.lax.scan(f_with_grad,(loss,0,nm_proj_w,w,grads),(activation_history_k,activation_history_i,activation_history_j))

    loss_train = loss[cfg.n_validation:].mean()
    loss_validation = loss[:cfg.n_validation].mean() if cfg.n_validation else 0

    return (loss_train, (loss_validation, w_new)), (None, grads)


# guided initialisation for modulatory projections.
# in the interest of computational efficiency, we assume that the weight updates do not affect future activation patterns at the pre-synaptic and modulating columns.
# note that this assumption may be violated when column m modulates a projection on a path from the input to column m.
def guided_weight_modification_m(rng_key,
                                 activation_history_k,
                                 activation_history_i,
                                 activation_history_j,
                                 w_geno,
                                 m_weights,
                                 w_record,
                                 fitness_per_task,
                                 new_projection,
                                 activation_function_i,
                                 activation_function_j):
    
    if cfg.guided_weight_modification_loss_locus == 'activation':
        # last weight update happens AFTER the final entry into the activation history, 
        # so for consistency with the activation history we should use the second-to-last weight.
        w_final = w_record[-2]
    
    if cfg.guided_weight_modification_loss_locus == 'weight':    
        w_final = w_record[-1]
    
    if (w_geno[None]==w_final).all():
        # this can happen when Rl updates were disabled and just got enabled during mutation (?)
        pprint('no weight change available for guided modification...')
        return None, None
    
    activation_history_k = jp.array(activation_history_k)
    activation_history_i = jp.array(activation_history_i)
    activation_history_j = jp.array(activation_history_j)
    
    pprint('guided_weight_modification_m with activation history:', activation_history_k.shape, jp.abs(activation_history_k).sum())
    if jp.isnan(activation_history_k).any():
        pprint('NaN value in activation history --> cancel guided_weight_init')
        return None, None
    n_a = activation_history_k.shape[0]
    
    # calculate fitness weight per task instance
    if cfg.enable_fitness_weighing:
        fmin = fitness_per_task.min()
        fmax = fitness_per_task.max()
        fitness_weights = (fitness_per_task-fmin)/(fmax-fmin)
        # clip fitness weights at random threshold and rescale to [0,1]
        if cfg.fitness_weight_clipping:
            rng_key, k = jax.random.split(rng_key)
            clip_threshold = jax.random.uniform(k)
            fitness_weights = (fitness_weights-clip_threshold)/(1-clip_threshold)
            fitness_weights = jp.clip(fitness_weights,0,1)
    else:
        fitness_weights = np.ones(cfg.n_task_instances)
        
    best_loss = np.inf
    best_m_weights = None
    stale = 0
    improvement_streak = 0
    n_updates_performed = 0
    
    if new_projection:
        rng_key, *kk = jax.random.split(rng_key,3)
        n_max_iterations = jax.random.randint(kk[0],(),minval=cfg.n_guided_weight_init_iteration_range[0],maxval=cfg.n_guided_weight_init_iteration_range[1])
        lr = jax.random.uniform(kk[1],(),minval=0,maxval=cfg.lr_guided_weight_init)
    else:
        rng_key, *kk = jax.random.split(rng_key,3)
        n_max_iterations = jax.random.randint(kk[0],(),minval=cfg.n_guided_weight_mutation_iteration_range[0],maxval=cfg.n_guided_weight_mutation_iteration_range[1])
        lr = jax.random.uniform(kk[1],(),minval=0,maxval=cfg.lr_guided_weight_mutation)

    for i_iteration in range(n_max_iterations):
        
        h = activation_history_k

        if cfg.cheap_GWM:
            (loss_train,(loss_validation,w_a_obtained)),grads = mGWM_sub_update(n_a,
                                                                                h,
                                                                                activation_history_i,
                                                                                activation_history_j,
                                                                                w_geno,
                                                                                w_final,
                                                                                m_weights,
                                                                                fitness_weights,
                                                                                activation_function_i,
                                                                                activation_function_j,
                                                                                lr)
        else:
            (loss_train,(loss_validation,w_a_obtained)),grads = guided_init_loss_grad(n_a,
                                                                            h,
                                                                            activation_history_i,
                                                                            activation_history_j,
                                                                            w_geno,
                                                                            w_final,
                                                                            m_weights,
                                                                            fitness_weights,
                                                                            activation_function_i,
                                                                            activation_function_j)

        if cfg.n_validation:
            loss = loss_validation
        else:
            loss = loss_train
        
        if i_iteration==0:
            initial_loss = loss

        if loss < best_loss:
            best_loss = loss
            best_m_weights = [w.copy() for w in m_weights]
            stale = 0
            improvement_streak += 1
            if improvement_streak%cfg.guided_drop_learning_rate_at_stale_count == 0:
                lr = min(1.5*lr,cfg.lr_guided_weight_init)
        else:
            stale += 1
            if stale%cfg.guided_drop_learning_rate_at_stale_count == 0:
                lr *= 0.5
            if stale == cfg.n_guided_weight_stop_at_stale_count:
                pprint('stale at iteration:', i_iteration)
                break
        w_geno_grads, m_w_grads = grads
        if cfg.optimise_geno_weights_in_guided_modification:
            w_geno -= lr*(jp.sign(w_geno_grads) if cfg.guided_weight_init_signSGD else w_geno_grads)
        for i_weight, g in enumerate(m_w_grads):
            m_weights[i_weight] -= lr*(jp.sign(g) if cfg.guided_weight_init_signSGD else g)
        n_updates_performed += 1
        utils.print_progress(i_iteration,n_max_iterations,message='NN:'+str(individual_id).rjust(3)+'  loss: '+str('%1.5e'%loss).rjust(11)+'  rate: '+str('%1.5e'%lr).rjust(11))
    if stale < cfg.n_guided_weight_stop_at_stale_count:
        utils.print_progress(message='NN:'+str(individual_id).rjust(3)+'  loss: '+str('%1.5e'%loss).rjust(11)+'  rate: '+str('%1.5e'%lr).rjust(11))
    
    w_final_std = jp.std(w_final,axis=0)
    pprint('weight range over w_final:', w_final.min(), w_final.max())
    pprint('standard deviation of weight over w_final:')
    pprint('  min:', w_final_std.min())
    pprint('  avg:', w_final_std.mean())
    pprint('  max:', w_final_std.max())
    
    w_a_obtained_std = jp.std(w_a_obtained,axis=0)
    pprint('weight range over obtained a-weight matrix:', w_a_obtained.min(), w_a_obtained.max())
    pprint('standard deviation of weight over obtained a-weight matrix:')
    pprint('  min:', w_a_obtained_std.min())
    pprint('  avg:', w_a_obtained_std.mean())
    pprint('  max:', w_a_obtained_std.max())
    
    w_a = jp.broadcast_to(w_geno,(cfg.n_task_instances,)+w_geno.shape)
    i_to = cfg.n_validation or None
    
    if cfg.guided_weight_modification_loss_locus == 'activation':
        
        n = cfg.n_trials_to_use_in_guided_init_loss_weights
        
        f = jax.vmap(dense_forward,in_axes=[None,0,None])
        aj_via_geno = f(w_geno,activation_history_i[-n:,:i_to],activation_function_i)
        f = jax.vmap(multi_w_dense_forward,in_axes=[None,0,None])
        aj_via_final = f(w_final[:i_to],activation_history_i[-n:,:i_to],activation_function_i)
        aj_new = f(w_a_obtained[:i_to],activation_history_i[-n:,:i_to],activation_function_i)
        
        if cfg.actvation_locus_loss_considers_activation_from_elsewhere:
            f = multi_multi_w_dense_forward
            aj_via_original = f(jp.array(w_record[-n-1:-1])[:,:i_to],activation_history_i[-n:,:i_to],activation_function_i)
            aj_from_elsewhere = activation_history_j[-n:,:i_to]-aj_via_original
            aj_via_geno += aj_from_elsewhere
            aj_via_final += aj_from_elsewhere
            aj_new += aj_from_elsewhere
        
        if cfg.actvation_locus_loss_applies_activation_functions:
            aj_via_geno = activation_function_j(aj_via_geno)
            aj_via_final = activation_function_j(aj_via_final)
            aj_new = activation_function_j(aj_new)
        
        d_original = (fitness_weights[None,:i_to,None]*jp.abs(aj_via_final-aj_via_geno)).mean()
        d_obtained = (fitness_weights[None,:i_to,None]*jp.abs(aj_via_final-aj_new)).mean()
        
    if cfg.guided_weight_modification_loss_locus == 'weight':
        d_original = (fitness_weights[:i_to,None,None]*jp.abs(w_final[:i_to]-w_a[:i_to])).mean()
        d_obtained = (fitness_weights[:i_to,None,None]*jp.abs(w_final[:i_to]-w_a_obtained[:i_to])).mean()
    
    if jp.isnan(d_obtained).any():
        pprint('guided modification failed with NaN --> discard')
        return None, None
    
    pprint('loss improvement:', initial_loss, '-->', best_loss, '(', n_updates_performed, '/', n_max_iterations, 'updates)')
    pprint('better than nothing at end?', d_obtained<d_original)
    pprint('d_original:', d_original, 'd_obtained:', d_obtained)
    
    # determine if the obtained weight matrix outperforms the genotypic weight matrix
    keep = d_obtained<d_original
    
    # if the obtained weight matrix was worse, we discard the result
    if not keep:
        pprint('no improvement --> discard')
        return None, None
    
    # if we are not printing analysis of the optimisation result, we are done here
    if cfg.suppress_nn_prints:
        return best_m_weights, w_geno
    
    # below this point we run a basic comparison between the original learning process and the learning process with the new weight matrix,
    # on the (false) assumption that the activation sequence on the pre-synaptic and modulating columns remain the same.
    # results are printed to the terminal only, so we can skip this if nn prints are suppressed.
    pprint('range of fitness_weights:', fitness_weights.min(), fitness_weights.max())
    pprint('number of non-zero fitness weights:', (fitness_weights>0).sum(), '/', fitness_weights.shape[0])
    
    initial_w_diff = (fitness_weights[:,None,None]*jp.abs(w_a-w_final)).mean()
    pprint('initial distance to w_final:', initial_w_diff)
    axes = np.arange(w_a.ndim)[1:]
    
    i = 1 
    for am, ai, aj in zip(activation_history_k[1:],activation_history_i[1:],activation_history_j[1:]):
        
        w_a, g = nm_weight_update(am,ai,aj,w_a,best_m_weights,w_geno)
        
        print_step = i<10 or (i+1)%int(cfg.n_trials_per_individual*cfg.n_trial_time_steps/10)==0 or i==n_a-1
        
        if cfg.guided_weight_modification_loss_locus == 'activation':
            
            aj_new = multi_w_dense_forward(w_a[:i_to],ai[:i_to],activation_function_i)
            aj_via_final = multi_w_dense_forward(w_final[:i_to],ai[:i_to],activation_function_i)
            
            if cfg.actvation_locus_loss_considers_activation_from_elsewhere:
                aj_original_local = multi_w_dense_forward(w_record[i-1][:i_to],ai[:i_to],activation_function_i)
                aj_from_elsewhere = aj[:i_to]-aj_original_local
                aj_new += aj_from_elsewhere
                aj_via_final += aj_from_elsewhere
            
            aj_original = aj[:i_to]    
            if cfg.actvation_locus_loss_applies_activation_functions:
                aj_new = activation_function_j(aj_new)
                aj_via_final = activation_function_j(aj_via_final)
                aj_original = activation_function_j(aj_original)
            
            if print_step:
                df = (fitness_weights[:i_to,None]*jp.abs(aj_new-aj_via_final)).mean(1)
                dh = (fitness_weights[:i_to,None]*jp.abs(aj_original-aj_via_final)).mean(1)

        if cfg.guided_weight_modification_loss_locus == 'weight':
            if print_step:
                df = (fitness_weights[:i_to,None,None]*jp.abs(w_a[:i_to]-w_final[:i_to])).mean(axes)
                dh = (fitness_weights[:i_to,None,None]*jp.abs(w_record[i][:i_to]-w_final[:i_to])).mean(axes)

        if print_step:
            df_mean = df.mean()
            dh_mean = dh.mean()
            pprint('step', i+1, 'g:', g.mean(), 'diff:', df_mean, 'history:', dh_mean, '=' if df_mean==dh_mean else ('o' if df_mean<dh_mean else 'x'))
            
        i += 1
        
    pprint('keep condition met')
    return best_m_weights, w_geno
        
    
# internal grad-compatible forward
@partial(jax.jit, static_argnums=(0,2))
def _forward(io_list_a,w_list_a,n_columns,obs):

    n_parallel = obs.shape[0]
    bias = jp.ones((n_parallel,1))
    obs = jp.concatenate((obs,bias),axis=-1)

    # initialise activation arrays
    a = [obs]+ \
        [np.zeros((n_parallel,cfg.hidden_dims+1)) for _ in range(n_columns-cfg.n_input_columns-cfg.n_output_columns)]+ \
        [np.zeros((n_parallel,1+1))]+ \
        [np.zeros((n_parallel,2*cfg.act_dims+1))]

    for ((i,j,_),w) in zip(io_list_a,w_list_a): # propagate in order of column index
        a[j] += multi_w_dense_forward(w,a[i],cfg.activation_function_obs if i<cfg.n_input_columns else cfg.activation_function_hidden)

    return a, cfg.activation_function_action(a[-1][:,:-1]), cfg.activation_function_value(a[-2][:,:-1]) # return activation in last two columns as action and V estimate

_forward_interval = jax.vmap(_forward,in_axes=[None,None,None,0])


# draw a random action from the given action distribution
@jax.jit
def draw_action(action_key,action_dist,action_sigma_bias):

    # split into mean and std parts
    loc = action_dist[...,:cfg.act_dims]
    scale = action_dist[...,cfg.act_dims:]+action_sigma_bias

    # draw action sample
    action = jax.lax.stop_gradient(loc+scale*jax.random.normal(action_key,loc.shape))

    return action, loc, scale
    

# RL loss function
@partial(jax.jit, static_argnums=(0,3,7,8))
def calculate_rl_loss(io_list_a,w_list_a,action_key,n_columns,obs,reward,action_sigma_bias,loss_weights,done):
    
    _, action_dist, V_estimate = _forward_interval(io_list_a,w_list_a,n_columns,obs)
    action_dist = action_dist[:-1]
    
    actions = []
    locs = []
    scales = []
    for t in range(cfg.n_trial_time_steps):
        action, loc, scale = draw_action(action_key[t],action_dist[t],action_sigma_bias)
        actions.append(action)
        locs.append(loc)
        scales.append(scale)
    action = jp.array(actions)
    loc = jp.array(locs)
    scale = jp.array(scales)
    
    # q value targets
    qval = jp.zeros(cfg.n_task_instances) if done else V_estimate[-1]
    qvals = []
    for t in range(reward.shape[0]-1,-1,-1):
        qval = reward[t]+cfg.rl_gamma*qval
        qvals = [qval]+qvals
    qvals = jp.array(qvals)
    
    # loss calculation
    advantage = qvals-jax.lax.stop_gradient(V_estimate[:-1,:,0])
    log_prob = jax.scipy.stats.norm.logpdf(action,loc=loc,scale=scale)
    loss_actor = -log_prob.mean(2)*advantage
    loss_critic = (V_estimate[:-1,:,0]-qvals)**2
    loss_entropy = -0.5*jp.log(2*jp.pi*scale)+0.5
    loss_entropy = loss_entropy.mean(2)
    
    # unpack loss weights
    actor_loss_weight, critic_loss_weight, entropy_loss_weight = loss_weights
    
    # compound loss
    loss = (actor_loss_weight*loss_actor.mean()+critic_loss_weight*loss_critic.mean()+entropy_loss_weight*loss_entropy.mean()).sum()
        
    return loss, (loss_actor,loss_critic, log_prob, V_estimate)


rl_loss_grad_v2 = jax.value_and_grad(
    calculate_rl_loss,
    argnums=1,
    has_aux=True)


# RL weight update logic
def _reinforcement_learning_update(io_list_a,
                                   w_list_a,
                                   action_key,
                                   n_columns,
                                   obs,
                                   reward,
                                   rl_learning_rate,
                                   action_sigma_bias,
                                   loss_weights,
                                   optimiser,
                                   opt_state,
                                   done):
            
    (loss,(loss_actor,loss_critic,log_prob,V_estimate)),grads = rl_loss_grad_v2(io_list_a,
                                                                                w_list_a,
                                                                                action_key,
                                                                                n_columns,
                                                                                obs,
                                                                                reward,
                                                                                action_sigma_bias,
                                                                                loss_weights,
                                                                                done)
    
    if cfg.clip_gradients:
        grads = clip_grad_norm(grads,cfg.grad_clip_norm)
    
    dw_list_a = []
    
    if optimiser is not None:
        # if using an external optimiser, let the optimiser compute the weight updates
        updates, opt_state = optimiser.update(grads,opt_state)
        for i, u in enumerate(updates):
            _, _, sus = io_list_a[i]
            w = w_list_a[i] + sus * u
            dw = w-w_list_a[i]
            dw_list_a.append(dw)
            w_list_a[i] = w
    else:
        # if not using an external optimiser, update following regular SGD or SignSGD
        for i, g in enumerate(grads):
            _, _, sus = io_list_a[i]
            w = w_list_a[i] - rl_learning_rate * sus * (log_prob.mean(1)[:,None,None]*jp.sign(g) if cfg.rl_signSGD else g)
            dw = w-w_list_a[i]
            dw_list_a.append(dw)
            w_list_a[i] = w
    
    return w_list_a, dw_list_a, V_estimate, grads, opt_state


@dataclass
class Projection:
    # genotypic weight matrix
    w: Union[list,np.ndarray]
    # (generation,individual) tuple indicating when the projection was created
    origin: tuple
    # (generation,individual) tuple indicating when the projection was last modified
    modified: tuple
    # priority for determining execution order of NM updates
    priority: float = 0
    # record of weights over learning process
    w_record: list = field(default_factory=list)
    
    
class nn:
    
    # initialises a new random individual.
    # random individuals are only generated while initialising the initial population, so generation index is set to 0.
    def __init__(self,rng_key,i_individual,peer_pipe):
        self.rng_key = rng_key
        self.peer_pipe = peer_pipe
        self.birth_generation = 0
        self.nm_enabled = cfg.nm_enabled
        self.rl_enabled = cfg.rl_enabled
        self.individual_id = i_individual
        self.n_columns = len(cfg.initial_connectivity)
        if cfg.random_connectivity_initialisation:
            self.connectivity_grid_a = (np.arange(self.n_columns)[:,None]<np.arange(self.n_columns)[None,:]).astype(int)
            rng_key, k = jax.random.split(rng_key)
            self.connectivity_grid_a *= jax.random.randint(k,self.connectivity_grid_a.shape,0,2)
            self.connectivity_grid_a = np.array(self.connectivity_grid_a)
            if cfg.disallow_direct:
                self.connectivity_grid_a[0,-cfg.n_output_columns:] = 0
        else:
            self.connectivity_grid_a = cfg.initial_connectivity
        self.connectivity_grid_m = np.zeros((self.n_columns,self.n_columns,self.n_columns))
        self.rl_susceptible = np.ones((self.n_columns,self.n_columns))*cfg.initial_rl_susceptibility
               
        # input layers have neuron count equal to extended observation size.
        # hidden layers all have neuron count equal to cfg.hidden_dims.
        # there are two output layers:
        #   - an output layer producing an estimate of state-value V (1 neuron).
        #   - an output layer producing an action distribution (neuron count equal to two times action size).
        # a bias neuron is added to each column. the activation value of the bias neuron is set to 1 during propagation.
        self.n_neurons_in_column = [cfg.obs_dims+1]+[cfg.hidden_dims+1]*(self.n_columns-3)+[1+1]+[2*cfg.act_dims+1]
        
        # RL-related genes
        if cfg.rl_learning_rate_init is None:
            self.rng_key, k = jax.random.split(self.rng_key)
            self.rl_learning_rate = jax.random.uniform(k,minval=cfg.rl_learning_rate_range[0],maxval=cfg.rl_learning_rate_range[1])
        else:
            self.rl_learning_rate = cfg.rl_learning_rate_init
            
        if cfg.action_sigma_bias_init is None:
            self.rng_key, k = jax.random.split(self.rng_key)
            self.action_sigma_bias = jax.random.uniform(k,minval=cfg.action_sigma_bias_range[0],maxval=cfg.action_sigma_bias_range[1])
        else:
            self.action_sigma_bias = cfg.action_sigma_bias_init
        
        # RL training loss weights
        if cfg.evolve_loss_weights:
            self.rng_key, *kk = jax.random.split(self.rng_key,4)
            self.actor_loss_weight = jax.random.uniform(kk[0],minval=0,maxval=cfg.max_actor_loss_weight)
            self.critic_loss_weight = jax.random.uniform(kk[1],minval=0,maxval=cfg.max_critic_loss_weight)
            self.entropy_loss_weight = jax.random.uniform(kk[2],minval=0,maxval=cfg.max_entropy_loss_weight)
        else:
            self.actor_loss_weight = cfg.actor_loss_weight
            self.critic_loss_weight = cfg.critic_loss_weight
            self.entropy_loss_weight = cfg.entropy_loss_weight
        
        # initialise projections
        self.init_geno_projections()
        self.mutations_applied = []
        self.generation_of_most_recent_mutation = None
        self.rl_buffer = {}
        self.rl_buffer['obs'] = []
        self.rl_buffer['reward'] = []
        self.rl_buffer['action_key'] = []
        self.reset()


    # sets NN ID as global variable to simplify printing with NN identification.
    def set_id_global(self):
        global individual_id
        individual_id = self.individual_id


    # initialise genotypic projections.
    # sets up a new genotype on basis of the connectivity grids.
    def init_geno_projections(self):
        def init_projection_type_a(connectivity_grid):
            geno_projections = np.empty(connectivity_grid.shape,dtype=object)
            if not connectivity_grid.any(): return geno_projections
            for i in range(self.n_columns):
                for j in range(self.n_columns):
                    if connectivity_grid[i,j]:
                        self.rng_key, k = jax.random.split(self.rng_key)
                        geno_projections[i,j] = make_projection_a(k,
                                                                  self.n_neurons_in_column[i],
                                                                  self.n_neurons_in_column[j],
                                                                  (self.birth_generation,self.individual_id))
                    else:
                        geno_projections[i,j] = None
            return geno_projections
        
        def init_projection_type_m(connectivity_grid):
            geno_projections = np.empty(connectivity_grid.shape,dtype=object)
            if not connectivity_grid.any(): return geno_projections
            pp = np.array(np.where(connectivity_grid)).T
            for m,i,j in pp:
                self.rng_key, k = jax.random.split(self.rng_key)
                geno_projections[m,i,j] = make_projection_m(k,
                                                            self.n_neurons_in_column[m],
                                                            self.n_neurons_in_column[i],
                                                            self.n_neurons_in_column[j],
                                                            (self.birth_generation,self.individual_id),
                                                            False)
            return geno_projections
            
        # init activation and modulation genotypic projections
        self.geno_projections_a = init_projection_type_a(self.connectivity_grid_a)
        self.geno_projections_m = init_projection_type_m(self.connectivity_grid_m)
        
        
    # builds the list of activation functions for all columns.
    def set_activation_functions(self):
        self.activation_functions = []
        for i in range(cfg.n_input_columns):
            self.activation_functions.append(cfg.activation_function_obs)
        for i in range(cfg.n_input_columns,self.n_columns-cfg.n_output_columns):
            self.activation_functions.append(cfg.activation_function_hidden)
        self.activation_functions.append(cfg.activation_function_action)
        self.activation_functions.append(cfg.activation_function_value)
        
        
    # initialise phenotypic projections.
    # produces a live NN from the genotype.
    def init_pheno_projections(self):
        
        # reset weight records.
        for indices in np.array(np.where(self.connectivity_grid_a)).T:
            self.geno_projections_a[tuple(indices)].w_record = []
        
        a_ij = np.array(np.where(self.connectivity_grid_a)).T
        m_kij = np.array(np.where(self.connectivity_grid_m)).T
        sus = self.rl_susceptible[a_ij[:,0],a_ij[:,1]]
        r = lambda v: np.repeat(v[None],cfg.n_task_instances,axis=0)
        
        self.column_has_input = np.concatenate((np.ones(cfg.n_input_columns),np.zeros(self.n_columns-cfg.n_input_columns)))
        for i, j in a_ij:
            if self.column_has_input[i]:
                self.column_has_input[j] = 1
                
        self.column_has_output = np.concatenate((np.zeros(self.n_columns-cfg.n_output_columns),np.ones(cfg.n_output_columns)))
        
        # determine which neurons affect output.
        # while loop is to ensure propagation of connectedness over nm projection (could be done more efficiently...).
        stable = False
        while not stable:
            stable = True
            for i, j in a_ij[::-1]:
                if self.column_has_output[j]:
                    if not self.column_has_output[i]:
                        self.column_has_output[i] = 1
                        stable = False
            for k, i, j in m_kij[::-1]:
                if self.column_has_input[k]:
                    if self.column_has_output[j]:
                        if not self.column_has_output[i]:
                            self.column_has_output[i] = 1
                            stable = False
        
        # construct io lists as tuples so they can be hashed (necessary for use as static args in jitted methods).
        self.io_list_a = tuple((i,j,s) for (i,j),s in zip(a_ij,sus) if self.column_has_input[i] and self.column_has_output[j])
        
        # construct activatory weight lists.
        self.w_list_a = [r(self.geno_projections_a[i,j].w) for (i,j,s) in self.io_list_a]
        grid_to_list_a = {(i,j):list_index for list_index,(i,j,_) in enumerate(self.io_list_a)}
        
        # lists for tracking per-projection total weight change induced by RL and NM.
        self.total_dw_list_rl = [0 for _ in self.io_list_a]
        self.total_dw_list_nm = [0 for _ in self.io_list_a]
        
        # sort indices by projection priority.
        m_kij = sorted(m_kij,key=lambda x: self.geno_projections_m[tuple(x)].priority)
        
        # construct modulatory weight lists.
        self.io_list_m = tuple((k,grid_to_list_a[(i,j)]) for (k,i,j) in m_kij if self.column_has_input[k] and self.column_has_input[i] and self.column_has_output[j] and (i,j) in grid_to_list_a)
        self.w_list_m = [self.geno_projections_m[k,i,j].w for (k,i,j) in m_kij if self.column_has_input[k] and self.column_has_input[i] and self.column_has_output[j] and (i,j) in grid_to_list_a]
        
        # mean RL learning rate over live projections.
        self.mean_active_rl_learning_rate = self.rl_learning_rate*np.mean([s for (i,j,s) in self.io_list_a])
            
    
    # convenience forward for external use
    def forward(self,obs):
        self.a, action0_dist, V_estimate0 = _forward(self.io_list_a,self.w_list_a,self.n_columns,obs)
        self.activation_history.append(self.a)
        return action0_dist, V_estimate0
    
    
    # processes observation and returns an action.
    def choose_action(self,action_key,obs):
        action_dist, v_est = self.forward(obs)
        action, loc, scale = draw_action(action_key,action_dist,self.action_sigma_bias)
        self.action_history.append((action,loc,scale))
        return action
    
    
    # updates the RL buffers and, if either a full RL update interval has passed or the trial has ended, run the RL weight update.
    def reinforcement_learning_update(self,action_key,obs0,obs1,reward,done):
        
        if not self.rl_enabled: return
        
        self.rl_buffer['obs'].append(obs0)
        self.rl_buffer['reward'].append(reward)
        self.rl_buffer['action_key'].append(action_key)
        
        if done or len(self.rl_buffer['reward']) == cfg.rl_weight_update_interval:
            self.rl_buffer['obs'].append(obs1)
            loss_weights = (self.actor_loss_weight, self.critic_loss_weight, self.entropy_loss_weight)
            self.w_list_a, dw_list_a, V_estimate0, grads, self.opt_state = \
                _reinforcement_learning_update(self.io_list_a,
                                               self.w_list_a,
                                               self.rl_buffer['action_key'],
                                               self.n_columns,
                                               jp.array(self.rl_buffer['obs']),
                                               jp.array(self.rl_buffer['reward']),
                                               self.rl_learning_rate,
                                               self.action_sigma_bias,
                                               loss_weights,
                                               self.optimiser,
                                               self.opt_state,
                                               done)
            self.rl_buffer['obs'] = []
            self.rl_buffer['reward'] = []
            self.rl_buffer['action_key'] = []
            
            for i, dw in enumerate(dw_list_a):
                self.total_dw_list_rl[i] += dw
    
    
    # run NM weight update.
    def neuromodulation_update(self):

        if not self.nm_enabled: return

        total_weight_change = 0
        for ((m,i_a),nm_proj_w) in zip(self.io_list_m,self.w_list_m): # propagate in order of column index
            i, j, _ = self.io_list_a[i_a]
            w_geno = self.geno_projections_a[i,j].w
            w, g = nm_weight_update(self.a[m],self.a[i],self.a[j],self.w_list_a[i_a],nm_proj_w,w_geno)
            dw = w-self.w_list_a[i_a]
            self.total_dw_list_nm[i_a] += dw
            self.w_list_a[i_a] = w
    
    
    # add the current weights to the weight record.
    def record_weights(self):
        for ((i,j,_),w) in zip(self.io_list_a,self.w_list_a):
            self.geno_projections_a[i,j].w_record.append(w.copy())
            
    
    # adds trial fitness to overall normalised fitness.
    def add_fitness(self,trial_fitness):
        self.raw_fitness += trial_fitness.mean()
        self.trial_fitness_log.append(trial_fitness)
        self.normalised_fitness = self.raw_fitness/len(self.trial_fitness_log)
    
    
    # report various statistics as a dict.
    # used by the main process to collect statistics.
    def report_stats(self):
        # calculate total net weight change due to RL
        self.total_weight_change_rl = 0
        for dw in self.total_dw_list_rl:
             self.total_weight_change_rl += jp.abs(dw).sum()
             
        # calculate total net weight change due to NM
        self.total_weight_change_nm = 0
        for dw in self.total_dw_list_nm:
             self.total_weight_change_nm += jp.abs(dw).sum()

        stats = {'normalised_fitness': self.normalised_fitness,
                 'total_weight_change_rl': self.total_weight_change_rl,
                 'total_weight_change_nm': self.total_weight_change_nm,
                 'trial_fitness_log': self.trial_fitness_log,
                 'mutations_applied': self.mutations_applied,
                 'mean_active_rl_learning_rate': self.mean_active_rl_learning_rate,
                 }
                 
        return stats
    
    
    # reports a record of the agent's actions.
    # used for trajectory visualisation by analyse_learning_process.py.
    def report_action_history(self):
        return np.array(self.action_history)
    
        
    # sends the genotype (plus some logs) to the NN at the other end of the given pipe.
    # the main process directs this pipe to an NN that should clone this NN.
    def send_genotype_to_child(self,pipe):
        genotype = {
            'parent_id': self.individual_id,
            'rl_learning_rate': self.rl_learning_rate,
            'action_sigma_bias': self.action_sigma_bias,
            'actor_loss_weight': self.actor_loss_weight,
            'critic_loss_weight': self.critic_loss_weight,
            'entropy_loss_weight': self.entropy_loss_weight,
            'connectivity_grid_a': self.connectivity_grid_a,
            'connectivity_grid_m': self.connectivity_grid_m,
            'rl_susceptible': self.rl_susceptible,
            'geno_projections_a': self.geno_projections_a,
            'geno_projections_m': self.geno_projections_m,
            'trial_fitness_log': self.trial_fitness_log, # not strictly genotype but used in mutation
            'activation_history': self.activation_history, # not strictly genotype but used in mutation
            }
        pipe.send(genotype)
    
    
    # rewrites the NN into a clone of the NN at the other end of the peer pipe (i.e. the parent).
    def clone_from_parent(self):
        
        genotype = self.peer_pipe.recv()
        
        # inherit learning genes
        self.rl_learning_rate = genotype['rl_learning_rate']
        self.action_sigma_bias = genotype['action_sigma_bias']
        self.actor_loss_weight = genotype['actor_loss_weight']
        self.critic_loss_weight = genotype['critic_loss_weight']
        self.entropy_loss_weight = genotype['entropy_loss_weight']
        
        # inherit connectivity grids
        self.connectivity_grid_a = genotype['connectivity_grid_a']
        self.connectivity_grid_m = genotype['connectivity_grid_m']
        
        self.rl_susceptible = genotype['rl_susceptible']
        
        # inherit weights
        for i in range(self.n_columns):
            for j in range(self.n_columns):
                p = genotype['geno_projections_a'][i,j]
                self.geno_projections_a[i,j] = p
                
        
        for m in range(self.n_columns):
            for i in range(self.n_columns):
                for j in range(self.n_columns):
                    p = genotype['geno_projections_m'][m,i,j]
                    self.geno_projections_m[m,i,j] = p
        
        self.trial_fitness_log = genotype['trial_fitness_log']
        self.activation_history = genotype['activation_history']
        
        
    # projection deletion mutation.
    def projection_deletion(self,connectivity_grid,geno_projections,mutation_rate,projection_type):
        local_changed = False
        self.rng_key, k = jax.random.split(self.rng_key)
        #pprint('connectivity grid:\n',connectivity_grid)
        if jax.random.uniform(k) < mutation_rate:
            if cfg.random_connectivity_initialisation or projection_type == 'M':
                pp = np.where(connectivity_grid)
            else:
                # don't allow deletion of initial connectivity
                pp = np.where(np.logical_and(connectivity_grid,1-cfg.initial_connectivity))
            if len(pp[0]): # abort if no projections left
                self.rng_key, k = jax.random.split(self.rng_key)
                indices = jax.random.choice(k,np.array(pp),axis=1)
                indices = tuple(indices)
                connectivity_grid[indices] = 0
                if projection_type == 'A':
                    if check_connected(connectivity_grid):
                        if len(indices)==2: # when disabling an activatory connection, we should also disable modulatory connections onto that connection
                            self.connectivity_grid_m[:,indices[0],indices[1]] = 0
                        # count deletion (deactivation) as modification of the projection (unsure about this)
                        geno_projections[indices] = None
                        local_changed = True
                        self.mutations_applied.append('projection_deletion_A')
                    else:
                        connectivity_grid[indices] = 1
                else:
                    local_changed = True
                    self.mutations_applied.append('projection_deletion_M')
        return connectivity_grid, geno_projections, local_changed
                       
    
    # mutation for inserting activatory projections.
    def projection_insertion_a(self,connectivity_grid,geno_projections,allow_recurrent):
        local_changed = False
        self.rng_key, k = jax.random.split(self.rng_key)
        if jax.random.uniform(k) < cfg.a_projection_insertion_mutation_rate:
            # select as presynaptic column a column that has input and is not an output column
            columns_with_input = jp.where(self.column_has_input[:-cfg.n_output_columns])[0]
            self.rng_key, k = jax.random.split(self.rng_key)
            i = jax.random.choice(k,columns_with_input)
            min_out = cfg.n_input_columns if allow_recurrent else i+1;
            out_available = min_out+jp.where(connectivity_grid[i][min_out:]==0)[0]
            if len(out_available):
                self.rng_key, k = jax.random.split(self.rng_key)
                j = jax.random.choice(k,out_available)
                cancel = cfg.disallow_direct and (i < cfg.n_input_columns and j >= self.n_columns-cfg.n_output_columns)
                if not cancel:
                    connectivity_grid[i,j] = 1
                    self.rng_key, k = jax.random.split(self.rng_key)
                    geno_projections[i,j] = make_projection_a(k,
                                                              self.n_neurons_in_column[i],
                                                              self.n_neurons_in_column[j],
                                                              (self.birth_generation,self.individual_id))
                    self.rng_key, k = jax.random.split(self.rng_key)
                    current_max_sus = max([sus for (_,_,sus) in self.io_list_a]) if cfg.initial_rl_susceptibility==0 else 1
                    self.rl_susceptible[i,j] = current_max_sus*jax.random.uniform(k)
                    local_changed = True
                    self.mutations_applied.append('projection_insertion_A[add]')
                
        return connectivity_grid, geno_projections, local_changed
    
    
    # mutation for inserting modulatory projections.
    def projection_insertion_m(self,connectivity_grid,geno_projections,allow_recurrent):
        local_changed = False
        self.rng_key, k = jax.random.split(self.rng_key)
        if jax.random.uniform(k) < cfg.m_projection_insertion_mutation_rate:
            pp = np.array(np.where(self.connectivity_grid_a==1))
            if len(pp[0]): # abort if connectivity matrix empty
                self.rng_key, k = jax.random.split(self.rng_key)
                i, j = jax.random.choice(k,pp,axis=1)
                columns_with_input = jp.where(self.column_has_input[:-cfg.n_output_columns])[0]
                cc = columns_with_input
                if len(cc):
                    for attempt in range(cfg.guided_m_insert_attempts):
                        if cfg.restrict_modulating_neurons_to_value_output:
                            m = self.n_columns-2 # value output neuron
                        else:
                            self.rng_key, k = jax.random.split(self.rng_key)
                            m = jax.random.choice(k,cc)
                        old_projection = geno_projections[m,i,j]
                        spec_string = '('+str(m)+' --> ('+str(i)+' --> '+str(j)+')'
                        w_record = self.geno_projections_a[i,j].w_record
                        if cfg.rl_enabled and cfg.use_guided_weight_init and self.rl_learning_rate>0 and self.rl_susceptible[i,j]>0 and len(w_record):
                            self.rng_key, k = jax.random.split(self.rng_key)
                            geno_projections[m,i,j] = make_projection_m(k,
                                                                        self.n_neurons_in_column[m],
                                                                        self.n_neurons_in_column[i],
                                                                        self.n_neurons_in_column[j],
                                                                        (self.birth_generation,self.individual_id),
                                                                        True)
                            ahk = [a[m] for a in self.activation_history]
                            ahi = [a[i] for a in self.activation_history]
                            ahj = [a[j] for a in self.activation_history]
                            w_geno_a = self.geno_projections_a[i,j].w
                            m_weights = self.geno_projections_m[m,i,j].w
                            fw = np.mean(self.trial_fitness_log[-cfg.n_trials_to_use_in_guided_init_loss_weights:],0)
                            afi = self.activation_functions[i]
                            afj = self.activation_functions[j]
                            self.rng_key, k = jax.random.split(self.rng_key)
                            w, w_geno_a = guided_weight_modification_m(k,ahk,ahi,ahj,w_geno_a,m_weights,w_record,fw,True,afi,afj)
                            if w is None:
                                connectivity_grid[m,i,j] = 0
                                geno_projections[m,i,j] = old_projection
                                pprint('[guided m-insert failure] columns: '+spec_string)
                            else:
                                if cfg.allow_only_one_nm_projection_per_a_projection:
                                    connectivity_grid[:,i,j] = 0
                                connectivity_grid[m,i,j] = 1
                                geno_projections[m,i,j].w = w
                                self.geno_projections_a[i,j].w = w_geno_a
                                local_changed = True
                                self.mutations_applied.append('projection_insertion_M_guided '+spec_string)
                                pprint('[guided m-insert success] columns: '+spec_string+' !!!')
                                break
                        else:
                            self.rng_key, k = jax.random.split(self.rng_key)
                            geno_projections[m,i,j] = make_projection_m(k,
                                                                        self.n_neurons_in_column[m],
                                                                        self.n_neurons_in_column[i],
                                                                        self.n_neurons_in_column[j],
                                                                        (self.birth_generation,self.individual_id),
                                                                        False)
                            pprint('[unguided m-insert] columns: '+spec_string)
                            local_changed = True
                            self.mutations_applied.append('projection_insertion_M_unguided '+spec_string)
                            break

        return connectivity_grid, geno_projections, local_changed
        

    # subroutine for modifying projection weights.
    def mutate_w(self,w):
        if type(w) is list:
            self.rng_key, k = jax.random.split(self.rng_key)
            i = jax.random.randint(k,(),0,len(w))
            w[i] = self.mutate_w(w[i])
            return w
        else:
            sh = w.shape
            self.rng_key, *kk = jax.random.split(self.rng_key,5)
            mask = jax.random.uniform(kk[0],sh)<jax.random.uniform(kk[1])
            dw = cfg.max_weight_mutation_strength*jax.random.uniform(kk[2],sh)*mask*make_projection_weights(kk[3],sh[0],np.prod(sh[1:]))
            return w+dw.reshape(sh)

    
    # method for mutating projection weights.
    def mutate_projection_weights(self,connectivity_grid,geno_projections,mutation_rate,projection_type):
        local_changed = False
        pp = np.where(connectivity_grid)
        if len(pp[0]): # abort if no projections exist
            self.rng_key, k = jax.random.split(self.rng_key)
            to_mutate = jp.where(jax.random.uniform(k,[len(pp[0])])<mutation_rate)[0]
            for i_mutate in to_mutate:
                indices = tuple(np.array(pp)[:,i_mutate])
                
                if projection_type == 'M': # = modulatory projection case
                    m, i, j = indices
                    i, j = int(i), int(j)
                    if self.geno_projections_a[i,j] is not None:
                        w_record = self.geno_projections_a[i,j].w_record
                        if cfg.rl_enabled and cfg.use_guided_weight_init and self.rl_learning_rate>0 and self.rl_susceptible[i,j]>0 and len(w_record):
                            self.rng_key, k = jax.random.split(self.rng_key)
                            if jax.random.uniform(k) < cfg.guided_weight_mutation_rate:
                                pprint('\n[mutate weights] columns:', m, '--> (', i, '-->', j,')')
                                ahk = [a[m] for a in self.activation_history]
                                ahi = [a[i] for a in self.activation_history]
                                ahj = [a[j] for a in self.activation_history]
                                w_geno_a = self.geno_projections_a[i,j].w
                                m_weights = self.geno_projections_m[m,i,j].w
                                fw = np.mean(self.trial_fitness_log[-cfg.n_trials_to_use_in_guided_init_loss_weights:],0)
                                afi = self.activation_functions[i]
                                afj = self.activation_functions[j]
                                self.rng_key, k = jax.random.split(self.rng_key)
                                w, w_geno_a = guided_weight_modification_m(k,ahk,ahi,ahj,w_geno_a,m_weights,w_record,fw,False,afi,afj)
                                if w is not None:
                                    geno_projections[m,i,j].w = w
                                    self.geno_projections_a[i,j].w = w_geno_a
                                    spec_string = '('+str(m)+' --> ('+str(i)+' --> '+str(j)+')'
                                    self.mutations_applied.append('weight_matrix_M_guided '+spec_string)
                                    local_changed = True
                                    pprint('[guided mutate weights completed] columns: '+spec_string)
                
                if projection_type == 'A': # = activatory projection case
                    if cfg.rl_enabled and cfg.use_guided_weight_init:
                        self.rng_key, k = jax.random.split(self.rng_key)
                        if jax.random.uniform(k) < cfg.guided_weight_mutation_rate:
                            i, j = indices
                            if self.geno_projections_a[i,j] is not None:
                                w_record = self.geno_projections_a[i,j].w_record
                                if len(w_record):
                                    self.rng_key, k = jax.random.split(self.rng_key)
                                    r = jax.random.uniform(k)
                                    w = r*w_record[-1].mean(0)+(1-r)*geno_projections[i,j].w
                                    geno_projections[i,j].w = w
                                    spec_string = '('+str(i)+' --> '+str(j)+')'
                                    pprint('guided A-weight matrix mutation applied '+spec_string)
                                    self.mutations_applied.append('weight_matrix_A_guided '+spec_string)
                                    local_changed = True
                    
                if not local_changed:
                    geno_projections[indices].w = self.mutate_w(geno_projections[indices].w)
                    self.mutations_applied.append('weight_matrix_'+projection_type+'_unguided')
                    local_changed = True
                
                geno_projections[indices].modified = (self.birth_generation,self.individual_id)
                    
        return geno_projections, local_changed
    
    
    # method for mutating neuromodulation projection execution priorities.
    def mutate_projection_priority(self,connectivity_grid,geno_projections):
        local_changed = False
        pp = np.where(connectivity_grid)
        if len(pp[0]): # abort if no projections exist
            self.rng_key, k = jax.random.split(self.rng_key)
            i_mu = jp.where(jax.random.uniform(k,[len(pp[0])])<cfg.priority_mutation_rate)[0]
            for i in i_mu:
                indices = tuple(np.array(pp)[:,i])
                self.rng_key, k = jax.random.split(self.rng_key)
                geno_projections[indices].priority = jax.random.uniform(k)
                pprint('priority mutation')
                self.mutations_applied.append('priority')
                local_changed = True
        return geno_projections, local_changed
    
    
    # mutate this individual.
    def mutate(self,i_generation):
        pprint('start mutation with RNG state:', self.rng_key)
        
        changed = False
        self.mutations_applied = []
        
        self.birth_generation = i_generation
        
        # mutate RL learning rate
        self.rng_key, k = jax.random.split(self.rng_key)
        if jax.random.uniform(k) < cfg.singleton_genes_mutation_rate:
            self.rng_key, k = jax.random.split(self.rng_key)
            self.rl_learning_rate += cfg.rl_learning_rate_mutation_strength*jax.random.normal(k,[1])[0]
            self.rl_learning_rate = np.clip(self.rl_learning_rate,cfg.rl_learning_rate_range[0],cfg.rl_learning_rate_range[1])
            changed = True
            self.mutations_applied.append('RL_learning_rate')
        
        # mutate action sigma bias
        self.rng_key, k = jax.random.split(self.rng_key)
        if jax.random.uniform(k) < cfg.singleton_genes_mutation_rate:
            self.rng_key, k = jax.random.split(self.rng_key)
            self.action_sigma_bias += cfg.action_sigma_bias_mutation_strength*jax.random.normal(k,[1])[0]
            self.action_sigma_bias = np.clip(self.action_sigma_bias,cfg.action_sigma_bias_range[0],cfg.action_sigma_bias_range[1])
            changed = True
            self.mutations_applied.append('action_sigma_bias')
            
        if cfg.evolve_loss_weights:
            # mutate actor loss weight
            self.rng_key, k = jax.random.split(self.rng_key)
            if jax.random.uniform(k) < cfg.singleton_genes_mutation_rate:
                self.rng_key, k = jax.random.split(self.rng_key)
                self.actor_loss_weight += 0.1*jax.random.normal(k)
                self.actor_loss_weight = np.clip(self.actor_loss_weight,0,2)
                changed = True
                self.mutations_applied.append('actor_loss_weight')
            
            # mutate critic loss weight
            self.rng_key, k = jax.random.split(self.rng_key)
            if jax.random.uniform(k) < cfg.singleton_genes_mutation_rate:
                self.rng_key, k = jax.random.split(self.rng_key)
                self.critic_loss_weight += 0.1*jax.random.normal(k)
                self.critic_loss_weight = np.clip(self.critic_loss_weight,0,2)
                changed = True
                self.mutations_applied.append('critic_loss_weight')
                
            # mutate entropy loss weight
            self.rng_key, k = jax.random.split(self.rng_key)
            if jax.random.uniform(k) < cfg.singleton_genes_mutation_rate:
                self.rng_key, k = jax.random.split(self.rng_key)
                self.entropy_loss_weight += 0.01*jax.random.normal(k)
                self.entropy_loss_weight = np.clip(self.entropy_loss_weight,0,0.1)
                changed = True
                self.mutations_applied.append('entropy_loss_weight')
        
        # projection deletion
        self.connectivity_grid_a, self.geno_projections_a, ch = self.projection_deletion(self.connectivity_grid_a,self.geno_projections_a,cfg.a_projection_deletion_mutation_rate,'A')
        changed = changed or ch
        self.connectivity_grid_m, self.geno_projections_m, ch = self.projection_deletion(self.connectivity_grid_m,self.geno_projections_m,cfg.m_projection_deletion_mutation_rate,'M')
        changed = changed or ch
        
        # mutate projection RL susceptibility
        pp = np.array(np.where(self.connectivity_grid_a))
        self.rng_key, k = jax.random.split(self.rng_key)
        to_mutate = jp.where(jax.random.uniform(k,[len(pp[0])])<cfg.projection_rl_susceptibility_mutation_rate)[0]
        for i_mutate in to_mutate:
            i, j = pp[:,i_mutate]
            self.rng_key, k = jax.random.split(self.rng_key)
            d = jax.random.uniform(k,minval=-cfg.rl_suscptibility_mutation_strength,maxval=cfg.rl_suscptibility_mutation_strength)
            self.rl_susceptible[i,j] = jp.clip(self.rl_susceptible[i,j]+d,0,1)
            changed = True
            self.mutations_applied.append('RL_suspectibility')
        
        # projection insertion
        self.connectivity_grid_a, self.geno_projections_a, ch = self.projection_insertion_a(self.connectivity_grid_a,self.geno_projections_a,False)
        changed = changed or ch
        if cfg.nm_enabled:
            self.connectivity_grid_m, self.geno_projections_m, ch = self.projection_insertion_m(self.connectivity_grid_m,self.geno_projections_m,True)
            changed = changed or ch
        
        # mutate projection weights
        prob_a = cfg.a_projection_weight_mutation_rate
        prob_m = cfg.m_projection_weight_mutation_rate

        self.geno_projections_a, ch = self.mutate_projection_weights(self.connectivity_grid_a,self.geno_projections_a,prob_a,'A')
        changed = changed or ch
        if cfg.nm_enabled:
            self.geno_projections_m, ch = self.mutate_projection_weights(self.connectivity_grid_m,self.geno_projections_m,prob_m,'M')
            changed = changed or ch
            self.geno_projections_m, ch = self.mutate_projection_priority(self.connectivity_grid_m,self.geno_projections_m)
        
        if len(self.mutations_applied):
            self.generation_of_most_recent_mutation = i_generation
        else:
            pprint('[mutation caused no apparent change in individual with id', self.individual_id, ' --> rerun mutation]')
            return self.mutate(i_generation)
        pprint('mutate ends with mutations_applied:', self.mutations_applied)
        return self.mutations_applied
        
    
    # reset NN to newborn state.
    def reset(self):
        self.init_pheno_projections()
        self.action_history = []
        self.raw_fitness = 0
        self.normalised_fitness = 0
        self.activation_history = []
        self.w_list_a_history = []
        self.total_weight_change_rl = 0
        self.total_weight_change_nm = 0
        self.trial_fitness_log = []
        self.trial_grads = []
        self.initialise_rl_optimiser()
        self.set_activation_functions()
        
        
    # initialise a fresh RL optimiser (if an explicit optimiser is specified).
    def initialise_rl_optimiser(self):
        if cfg.rl_optimiser is not None:
            if cfg.rl_optimiser is optax.sgd:
                self.optimiser = cfg.rl_optimiser(self.rl_learning_rate)
            else:
                self.optimiser = cfg.rl_optimiser(self.rl_learning_rate,
                                                eps=1e-8,
                                                decay=0.99,
                                                initial_scale=0,
                                                centered=False,
                                                )
            self.opt_state = self.optimiser.init(self.w_list_a)
        else:
            self.optimiser = None
            self.opt_state = None
        
    
    # reports mean magnitude of activatory and modulatory weights, in string format.
    # used by the main process to print statistics during evolution.
    def report_weight_stats(self):
        a_sum_size = 0
        a_sum_weight = 0
        for w in self.w_list_a:
            a_sum_size += w.size
            a_sum_weight += jp.abs(w).sum()
        if a_sum_size:
            a_sum_weight /= a_sum_size
        m_sum_size = 0
        m_sum_weight = 0
        for ww in self.w_list_m:
            m_sum_size += np.sum([w.size for w in ww])
            m_sum_weight += np.sum([jp.abs(w).sum() for w in ww])
        if m_sum_size:
            m_sum_weight /= m_sum_size
        return 'a:'+str(a_sum_weight)+' m:'+str(m_sum_weight)
        
