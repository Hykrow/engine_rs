use crate::tensor::Tensor;
use crate::utils::inits::kaiming;
use crate::{nn::layers::bind::ParamCursor, trace::NodeId};
use crate::trace::Trace;
use crate::ops::add; 
use crate::ops::matmul; 
pub struct Linear{
    pub w: NodeId,
    pub b: NodeId,
}

impl Linear{
    pub fn bind(cur: &mut ParamCursor) -> Linear{
        let (w, b) = cur.take2(); 
        Linear{w, b}
    }

    // x.w + b
    pub fn apply(&self, tr: &mut Trace, x: NodeId) -> NodeId{
        let x_dot_w = matmul(tr, x, self.w); 
        add(tr, x_dot_w, self.b)
    }

    pub fn init_kaiming(in_dim: usize, out_dim: usize)-> Vec<Tensor>{
        vec![kaiming(in_dim, out_dim), Tensor::zeros(&[out_dim])]

    }
}

