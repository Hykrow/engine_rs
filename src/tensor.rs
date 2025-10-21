use std::ops::{Add, Mul};

#[derive(Debug)]
pub struct Tensor {
    pub val: Vec<f32>,
    pub shape : Vec<usize>, 
    pub strides : Vec<usize> 
}





pub struct Shape(Vec<usize>);

impl Tensor{
    pub fn compute_strides(shape : &[usize]) -> Vec<usize>{
        let mut strides = vec![0; shape.len()];
        let mut product = 1; 
        for (i, dim) in shape.iter().rev().enumerate(){
            strides[i] = product; 
            product*=dim;
        }
        strides
    }
    pub fn ones(shape : &[usize])-> Tensor{
        Tensor{
            val: vec![1.0; shape.iter().product()], 
            shape : shape.to_vec(), 
            strides : Self::compute_strides(shape)
        }
    }
}

