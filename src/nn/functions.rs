use smallvec::SmallVec;

use crate::tensor::Tensor; 
use crate::tensor::Numel;
use crate::trace::{Trace, NodeId, Node};
use crate::ops::hadamard_mul_direct;

use std::result;
use std::sync::Arc; 
use std::ops::Add;
use smallvec::smallvec;

pub fn apply<F>(tr: &mut Trace, a_id: NodeId, f_apply: F, f_backwards: fn(f32) -> f32) -> NodeId
    where 
    F: Fn(f32) -> f32, 

{
    let a= tr.get_tensor(a_id).clone();
    let c = a.apply(f_apply);
    
    
    let vjp = move |g_out: &Tensor| -> SmallVec<[(NodeId, Tensor); 2]>{
        smallvec![(a_id, hadamard_mul_direct(&a.apply(f_backwards), g_out).sum_over_broadcasted_batches(&a.shape))] 

    }; 

    tr.push(crate::trace::Node { value: c, parents_id: smallvec![a_id], vjp: Some(Box::new(vjp)), is_param: false })
}

pub fn tanh(tr: &mut Trace, a_id: NodeId) -> NodeId{
    apply(tr, a_id, |x| x.tanh(), |x| 1f32-x.tanh()*x.tanh())
}

pub fn relu(tr: &mut Trace, a_id: NodeId) -> NodeId{
    apply(tr, a_id, |x| x.max(0f32), |x| if x >= 0f32 {1f32} else{0f32})
}