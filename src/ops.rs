use crate::tensor::{self, Tensor};
use std::sync::Arc; 
use std::ops::{Add, Mul};
#[derive(Debug)]
pub enum Op {
    Add,
    MatMul, // two last shapes
    Tanh,
}



impl Op{
    pub fn forward(&self, parents : &[&Tensor]) -> Tensor{
        match self{
            Op::Add => parents[0] + parents[1],


        }

    }
    pub fn take_grad(&self, parents : &[Arc<Tensor>]) -> Tensor{

    }
}





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

    let (m, n) = (a.shape[0], a.shape[1]); // A: (m, n)
    let (n2, p) = (b.shape[0], b.shape[1]); // B: (n, p)
    assert_eq!(n, n2, "matmul2d: dimensions incompatibles: A(m,n)={},{} vs B(n,p)={},{}", n, m, n2, p);



    let mut c = Tensor::zeros(&[n, m]);
    for i in 0..m{
        for j in 0..p{
            let mut sum = 0.0;
            for k in 0..n{
                sum+=a.get2(i, k)*b.get2(k, j); 
            }
            c.set2(i, j, sum);
        }
    }
    c
}

fn tensor_mul_helper(a : &Tensor, b: &Tensor) -> Tensor{
    let n = a.shape.len(); 
    let m = b.shape.len();

    let access_shape_a: Vec<usize> = a.shape;


    assert_eq!(a.shape[n], b.shape[m-1], "dimensions non ok pour multiplication des tenseurs (2 dernieres couches)");
    let l = a.shape[n];

}

fn tensor_mul(a:  & Tensor, b: & Tensor) -> Tensor{
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


