use crate::tensor::{self, Tensor};
use std::sync::Arc; 
use crate::tensor::Numel; 
use std::ops::{Add, Mul};
#[derive(Debug)]
pub enum Op {
    Add,
    MatMul, // two last shapes
    Tanh,
}


/*
impl Op{
    pub fn forward(&self, parents : &[&Tensor]) -> Tensor{
        match self{
            Op::Add => parents[0] + parents[1],


        }

    }
    pub fn take_grad(&self, parents : &[Arc<Tensor>]) -> Tensor{

    }
}

*/



impl Add for &Tensor{
    type Output = Tensor; 

    fn add(self, other: &Tensor) -> Tensor{
        assert_eq!(self.shape, other.shape);
        let sum: Vec<f32> = self.data.iter().zip(other.data.iter()).map(|(a, b)|a+b).collect();
        Tensor::from_vec(&sum, &self.shape).unwrap()
    }
}






fn matmul2d(a: &Tensor, b: &Tensor) -> Tensor{
    assert_eq!(a.shape.len(), 2, "matmul2d: A doit être 2D, shape={:?}", a.shape);
    assert_eq!(b.shape.len(), 2, "matmul2d: B doit être 2D, shape={:?}", b.shape);

    let (m, p) = (a.shape[0], a.shape[1]); // A: (m, n)
    let (p2, n) = (b.shape[0], b.shape[1]); // B: (n, p)
    assert_eq!(p, p2, "matmul2d: dimensions incompatibles");



    let mut c = Tensor::zeros(&[m, n]);
    for i in 0..m{
        for j in 0..n{
            let mut sum = 0.0;
            for k in 0..p{
                sum+=a.get2(i, k)*b.get2(k, j); 
            }
            c.set2(i, j, sum);
        }
    }
    c
}
#[inline(always)]
fn matmul2d_vec(a: &Tensor, b: &Tensor) -> Vec<f32>{
    assert_eq!(a.shape.len(), 2, "matmul2d: A doit être 2D, shape={:?}", a.shape);
    assert_eq!(b.shape.len(), 2, "matmul2d: B doit être 2D, shape={:?}", b.shape);

    let (m, p) = (a.shape[0], a.shape[1]); // A: (m, n)
    let (p2, n) = (b.shape[0], b.shape[1]); // B: (n, p)
    assert_eq!(p, p2, "matmul2d: dimensions incompatibles");



    let mut c = Vec::with_capacity(m*n);
    for i in 0..m{
        for j in 0..n{
            let mut sum = 0.0;
            for k in 0..p{
                sum+=a.get2(i, k)*b.get2(k, j); 
            }
            c.push(sum);
        }
    }
    c
}
fn tensor_mul_helper(a : &Tensor, b: &Tensor) -> Tensor{
    let a_order = a.shape.len(); 
    let b_order = b.shape.len();


    let m = a.shape[a_order-2];
    let n = b.shape[b_order-1];
    let p = a.shape[a_order-1];
    assert_eq!(a.shape[a_order-1], b.shape[b_order-2], "dimensions non ok pour multiplication des tenseurs (2 dernieres couches)");
    

    let batch_a = &a.shape[0..a_order-2];
    let batch_b = &b.shape[0..b_order-2]; 



    let batch = Tensor::broadcast_shape(&batch_a, &batch_b).unwrap();

    let mut out_shape = batch.clone(); 
    out_shape.push(m); 
    out_shape.push(n);
    println!("Tensor a : {}", a); 
    println!("Tensor b : {}", b);
    println!("out shape : {:?}", out_shape);
    let a_b = a.broadcast_view(&[batch.clone(), vec![m, p]].concat()).unwrap();
    let b_b = b.broadcast_view(&[batch.clone(), vec![p, n]].concat()).unwrap();

    println!("broadcasted succesfully! Tensor a_b : {}, \n Tensor b_b : {}", a_b, b_b);

    let mut c = Vec::with_capacity(out_shape.numel());

    let lin_max = batch.numel();
    let mut a_offset = 0; 
    let mut b_offset = 0; 
    println!("{}", lin_max);
    for lin in 0..lin_max{ // iterer a travers les batch
        let idx_from_lin = Tensor::idx_from_lin(&batch, lin);
        let a_b_2d = a_b.vue2d(a_b.batch_offset(&idx_from_lin));
        let b_b_2d = b_b.vue2d(b_b.batch_offset(&idx_from_lin));
        println!("A vue 2d: {} \n, B vue 2d: {}, \n, lin: {}", a_b_2d, b_b_2d, lin);
        let c_b_2d = matmul2d_vec(&a_b_2d, &b_b_2d);
        for el in c_b_2d{
            c.push(el);
        }
        

    }
    Tensor::from_vec(&c, &out_shape).unwrap()

}

pub fn tensor_mul(a:  & Tensor, b: & Tensor) -> Tensor{
    let len_a = a.shape.len(); 
    let len_b = b.shape.len();
    
    // dimensions ok pour multiplier 2 derniers en mode matrice.. 
    match(len_a, len_b){
        (1, 1)=> {
            let unsqueezed_a = a.unsqueeze_view(0);
            let unsqueezed_b =b.unsqueeze_view(1);
            matmul2d(&unsqueezed_a, &unsqueezed_b)
            
        }
        (_, 1)=>{
            let unsqueezed_b = b.unsqueeze_view(1);
            tensor_mul_helper(a, &unsqueezed_b)
            
        }
        (1, _)=>{
            let unsqueezed_a = a.unsqueeze_view(0);
            tensor_mul_helper(&unsqueezed_a, b)
            
        }
        (_, _)=>tensor_mul_helper(a, b)
    }
    
}


