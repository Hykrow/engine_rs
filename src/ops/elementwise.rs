use smallvec::SmallVec;

use crate::tensor::Tensor; 
use crate::tensor::Numel;
use crate::trace::{Trace, NodeId, Node};
use std::sync::Arc; 
use std::ops::{Add, Sub, Div, Mul};
use smallvec::smallvec;


pub fn hadamard_mul_direct(a: &Tensor, b: &Tensor ) -> Tensor{

    let out_shape = Tensor::broadcast_shape(&a.shape, &b.shape).unwrap();
 //   println!("a: {}, b: {} out_shape: {:?}", a, b, out_shape);
    let a_broad = a.broadcast_view(&out_shape).unwrap();
    let b_broad = b.broadcast_view(&out_shape).unwrap();
 //   println!("a broad: {} \n b broad : {}", a_broad, b_broad);
    let mut c = Vec::with_capacity(out_shape.numel());
    for lin in 0..out_shape.numel(){
        c.push(a_broad.get_from_lin(lin)*b_broad.get_from_lin(lin));
    }
    
    Tensor::new(Arc::new(c), &out_shape, 0)
}

pub fn hadamard_mul(tr: &mut Trace, a: NodeId, b: NodeId) -> NodeId{
    let va = tr.get_tensor(a).clone();
    let vb = tr.get_tensor(b).clone();



    let result_product = hadamard_mul_direct(&va, &vb);

    let vjp = move |g_out: &Tensor| -> SmallVec<[(NodeId, Tensor); 2]>{
        let ga = hadamard_mul_direct(&g_out, &vb).sum_over_broadcasted_batches(&va.shape);
        let gb = hadamard_mul_direct(&g_out, &va).sum_over_broadcasted_batches(&vb.shape);
        smallvec![(a, ga), (b, gb)]

    }; 

    tr.push(Node { value: result_product, parents_id: smallvec![a, b], vjp: Some(Box::new(vjp)), is_param: false })
}

pub fn add(tr: &mut Trace, a: NodeId, b: NodeId) -> NodeId{
    let va = tr.get_tensor(a).clone();
    let vb = tr.get_tensor(b).clone(); 

    let res = &va+&vb; 
    let vjp = move |g_out: &Tensor| -> SmallVec<[(NodeId, Tensor); 2]>{
        let ga = g_out.sum_over_broadcasted_batches(&va.shape); // TODO: check si bien ok de faire ca sur ref vs sur non ref (& vs non &)
        let gb = g_out.sum_over_broadcasted_batches(&vb.shape);

        smallvec![(a, ga), (b, gb)]
    };
    tr.push(Node { value: res, parents_id: smallvec![a, b], vjp: Some(Box::new(vjp)), is_param: false })
    
}


pub fn sub(tr: &mut Trace, a: NodeId, b: NodeId) -> NodeId{ // TODO: check si c'est le bon endroit ou multiplier par moins 1
    let va = tr.get_tensor(a).clone();
    let vb = tr.get_tensor(b).clone(); 

    let res = &va-&vb; 
    let vjp = move |g_out: &Tensor| -> SmallVec<[(NodeId, Tensor); 2]>{
        let ga = g_out.sum_over_broadcasted_batches(&va.shape); // TODO: check si bien ok de faire ca sur ref vs sur non ref (& vs non &)
        let gb = g_out.sum_over_broadcasted_batches(&vb.shape).apply(|x| x*(-1f32));

        smallvec![(a, ga), (b, gb)]
    };
    tr.push(Node { value: res, parents_id: smallvec![a, b], vjp: Some(Box::new(vjp)), is_param: false })
    
}

impl Add for &Tensor{
    type Output = Tensor;
    fn add(self, b: &Tensor) -> Tensor{

        let broadcast_shape = Tensor::broadcast_shape(&self.shape, &b.shape).unwrap();

        
        
        let sz = broadcast_shape.numel();
        let mut vec = Vec::with_capacity(sz);       
        let a_b = self.broadcast_view(&broadcast_shape).unwrap();
        let b_b = b.broadcast_view(&broadcast_shape).unwrap();


       // println!("shape vers qui broadcast : {:?} \n, a_b: {}\n, b_b : {},\n a: {a}")
        for lin in 0..sz{
            vec.push(a_b.get_from_lin(lin)+ b_b.get_from_lin(lin));
            
        }
        Tensor { data: Arc::new(vec), shape: broadcast_shape, strides: a_b.strides, offset: 0 }
    }
}



impl Sub for &Tensor{
    type Output = Tensor;
    fn sub(self, b: &Tensor) -> Tensor{

        self+ &b.apply(|x| x*-1f32)
    }
}



impl Div for &Tensor{
    type Output = Tensor;
    fn div(self, b: &Tensor) -> Tensor{
        hadamard_mul_direct(self, &b.apply(|x| 1f32/x))
    }
}

impl Mul for &Tensor{
    type Output = Tensor;
    fn mul(self, b: &Tensor) -> Tensor{
        hadamard_mul_direct(self, b)
    }
}



