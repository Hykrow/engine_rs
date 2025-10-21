use crate::tensor::Tensor;
use std::sync::Arc; 

#[derive(Debug)]
pub enum Op {
    Add,
    MatMul, // two last shapes
    EigSum, 
    Tanh,
}



impl Op{
    pub fn apply(&self, parents : &[&Tensor]) -> Tensor{
        match self{
            Op::Add => Tensor{val: parents[0].val + parents[1].val}, 
            Op::Tanh => Tensor{val: parents[0].val},
     


        }

    }
    pub fn take_grad(&self, parents : &[Arc<Tensor>]) -> Tensor{
        Op::Add => Tensor{},
        Op::Mul => {
            let product = parents.iter().copied().product();
            parents.map()
        }
    }
}