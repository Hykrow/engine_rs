use crate::trace::Trace;
use crate::tensor::Tensor;


pub fn get_params_id(tr: &mut Trace, params: &[Tensor])-> Vec<usize>{
    let mut param_ids = Vec::with_capacity(params.len()); 

    // ca permet d'avoir leur id. 
    for p in params{
        param_ids.push(tr.param(p.clone()));
    }
    param_ids
}