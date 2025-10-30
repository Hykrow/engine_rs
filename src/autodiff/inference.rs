use crate::tensor::Tensor; 
use crate::trace::{Trace, NodeId}; 

pub fn inference(
    params: &[Tensor], 
    build: impl Fn (&mut Trace, &[NodeId]) -> NodeId, 

) -> Tensor {
    let mut tr = Trace::new();

    let mut param_ids = Vec::with_capacity(params.len()); 

    // ca permet d'avoir leur id. 
    for p in params{
        param_ids.push(tr.param(p.clone()));
    }

    println!("params id: {:?} ", param_ids);


    let pred_id = build(&mut tr, &param_ids);

    println!("successfully built");
    let pred = tr.get_tensor(pred_id).clone(); 


    pred
}