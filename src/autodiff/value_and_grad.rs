use crate::tensor::Tensor; 
use crate::trace::{Trace, NodeId}; 

pub fn value_and_grad(
    params: &[Tensor], 
    build: impl Fn (&mut Trace, &[NodeId]) -> NodeId, 

) -> (Tensor, Vec<Tensor>) {
    let mut tr = Trace::new();

    let mut param_ids = Vec::with_capacity(params.len()); 

    // ca permet d'avoir leur id. 
    for p in params{
        param_ids.push(tr.param(p.clone()));
    }

    println!("params id: {:?} ", param_ids);


    let loss_id = build(&mut tr, &param_ids);

    println!("successfully built");
    let loss_val = tr.get_tensor(loss_id).clone(); 

    let grads = tr.backward_param_grads(loss_id);

    (loss_val, grads)
}