use crate::tensor::{self, Tensor};
use core::panic;
use std::sync::Arc; 
use crate::tensor::Numel; 
use std::ops::{Add, Mul};
#[derive(Debug)]
pub enum Op {
    Add,
    MatMul, // two last shapes
    Tanh,
    ReLU
}



impl Op{
    pub fn forward(&self, parents : &[&Tensor]) -> Tensor{
        match self{
            Op::Add => parents[0] + parents[1],
            Op::MatMul => tensor_mul(parents[0], parents[1]),
            Op::Tanh => parents[0].apply(f32::tanh),
            Op::ReLU => parents[0].apply(|x| x.max(0f32)),
 
        }

    }
    //pub fn take_grad(&self, parents : &[Arc<Tensor>]) -> Tensor{

    //}

    pub fn backward(&self, parents: &[&Tensor], tensor: &Tensor, before_grad: &Tensor) -> Vec<Tensor>{

        match self{
            Op::Add =>{
                 
                vec![before_grad.sum_over_broadcasted_batches(&parents[0].shape), before_grad.sum_over_broadcasted_batches(&parents[1].shape)] // a check si on peut faire sans clone. En vrai on modifiie jamais before_grad..
            }, 
            Op::MatMul =>{
                println!("Matrice a: {}, \n, Matrice b {} , \n BEFORE GRAD:  {}", parents[0], parents[1], before_grad);

                let grad_left = tensor_mul(before_grad, &parents[1].mat_transpose()).sum_over_broadcasted_batches(&parents[0].shape);
                let grad_right = tensor_mul(&parents[0].mat_transpose(), &before_grad).sum_over_broadcasted_batches(&parents[1].shape);
                vec![grad_left, grad_right]
            }, 
            Op::Tanh=>{
     
                vec![hadamard_mul(&tensor.apply(|y| 1f32-y*y), before_grad)] 
            }, 
            Op::ReLU=>{
                vec![hadamard_mul(&tensor.apply(|y| if y > 0f32 {1f32} else {0f32}), before_grad)]
            }

        }
    }
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
        // hack pour que ce soit plus simple : 
        let a_order = a.shape.len(); 
        let b_order = b.shape.len();
        assert_eq!(a.shape[a_order-1], b.shape[b_order-2], "dimensions non ok ");

        let m = a.shape[a_order-2];
        let n = b.shape[b_order-1];
        //TODO : gérer le fait que il faut peut etre broadcast sur les bord s ? AUCUN sens de faire squeeze view puis unsqueeze view ? CAR ON oublie le fait que c du roadcast
        
        let p = a.shape[a_order-1];
        println!("Tensor a : {}", a); 
        println!("Tensor b : {}", b);
        

        let batch_a = &a.shape[0..a_order-2];
        let batch_b = &b.shape[0..b_order-2]; 



        let batch = Tensor::broadcast_shape(&batch_a, &batch_b).unwrap();

        let mut out_shape = batch.clone(); 
        out_shape.push(m); 
        out_shape.push(n);
        println!("out shape : {:?} DIMENSIONS : m {} n {} p {}", out_shape,m, n, p);

        let a_b = a.broadcast_view(&[batch.clone(), vec![m, p]].concat()).unwrap();
        let b_b = b.broadcast_view(&[batch.clone(), vec![p, n]].concat()).unwrap();
        
        
        
        

        println!("broadcasted succesfully! Tensor a_b : {}, \n Tensor b_b : {}", a_b, b_b);
       // assert_eq!(a_b.shape[a_order-1], b_b.shape[b_order-2], "dimensions non ok pour multiplication des tenseurs (2 dernieres couches)");

        let mut c = Vec::with_capacity(out_shape.numel());

        let lin_max = batch.numel();
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

    pub fn tensor_mul(a:  & Tensor, b: & Tensor) -> Tensor{ // TODO: NE PAS MATCH SUR LES LEN CAR PAS CORRECT. IMAGINE : (N, 784) POUR MINST : CEST BIEN UN BATCH DE N VECTEURS DE TAILLE 784 MAIS LA CA SERA TRAIT
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



