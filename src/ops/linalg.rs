

use smallvec::{smallvec, SmallVec};

use crate::tensor::Tensor; 
use crate::tensor::Numel;
use crate::trace::{Trace, NodeId};


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

   // println!("TENSOR A : {} \n TENSOR B: {}", a, b);
    assert_eq!(a.shape[a_order-1], b.shape[b_order-2], "dimensions non ok; affichage des tenseurs : Tenseur a: {} \n Tenseur b: {} ", a, b);

    let m = a.shape[a_order-2];
    let n = b.shape[b_order-1];
    //TODO : gérer le fait que il faut peut etre broadcast sur les bord s ? AUCUN sens de faire squeeze view puis unsqueeze view ? CAR ON oublie le fait que c du roadcast
    
    let p = a.shape[a_order-1];
  //  println!("Tensor a : {}", a); 
  //  println!("Tensor b : {}", b);
    

    let batch_a = &a.shape[0..a_order-2];
    let batch_b = &b.shape[0..b_order-2]; 



    let batch = Tensor::broadcast_shape(&batch_a, &batch_b).unwrap();

    let mut out_shape = batch.clone(); 
    out_shape.push(m); 
    out_shape.push(n);
  //  println!("out shape : {:?} DIMENSIONS : m {} n {} p {}", out_shape,m, n, p);

    let a_b = a.broadcast_view(&[batch.clone(), vec![m, p]].concat()).unwrap();
    let b_b = b.broadcast_view(&[batch.clone(), vec![p, n]].concat()).unwrap();
    
    
    
    

  //  println!("broadcasted succesfully! Tensor a_b : {}, \n Tensor b_b : {}", a_b, b_b);
    // assert_eq!(a_b.shape[a_order-1], b_b.shape[b_order-2], "dimensions non ok pour multiplication des tenseurs (2 dernieres couches)");

    let mut c = Vec::with_capacity(out_shape.numel());

    let lin_max = batch.numel();
   // println!("{}", lin_max);
    for lin in 0..lin_max{ // iterer a travers les batch
        let idx_from_lin = Tensor::idx_from_lin(&batch, lin);
        let a_b_2d = a_b.vue2d(a_b.batch_offset(&idx_from_lin));
        let b_b_2d = b_b.vue2d(b_b.batch_offset(&idx_from_lin));
      //  println!("A vue 2d: {} \n, B vue 2d: {}, \n, lin: {}", a_b_2d, b_b_2d, lin);
        c.extend(matmul2d_vec(&a_b_2d, &b_b_2d));
 
    }
    Tensor::from_vec(&c, &out_shape).unwrap()

}

pub fn tensor_mul(a:  &Tensor, b : &Tensor) -> Tensor{ // TODO: NE PAS MATCH SUR LES LEN CAR PAS CORRECT. IMAGINE : (N, 784) POUR MINST : CEST BIEN UN BATCH DE N VECTEURS DE TAILLE 784 MAIS LA CA SERA TRAITE COMME UNE MATRICE
    let len_a = a.shape.len();  // TODO: CHECK LES SQUEEZES OU NON. PEUT ETRE OK GRACE AU SUM ON BROADCASTED MAIS PAS TROP SUR
    let len_b = b.shape.len();
    // dimensions ok pour multiplier 2 derniers en mode matrice..  

    
    //TODO: CHECK LES UNSQUEEZE VIEW VS SQUEEZE VIEW
    match(len_a, len_b){
        (0, 0)=>{
            a.apply(|x| x*b.data[0])
        }
        (0, 1)=>{
            b.apply(|x| x*a.data[0])
        }
        (1, 0)=>{ // b est un scalaire
            a.apply(|x| x*b.data[0])
        }
        (1, 1)=> {
            let unsqueezed_a = a.unsqueeze_view(0);
            let unsqueezed_b =b.unsqueeze_view(1);
            // on a donc (1, m) (m, 1) => (1, 1)
            tensor_mul_helper(&unsqueezed_a, &unsqueezed_b).squeeze_view(0).squeeze_view(0) // ()
            
        }
        (_, 1)=>{
                                                        // a =>    (..., N, M)
            let unsqueezed_b = b.unsqueeze_view(1);// (M, 1)
            let v = tensor_mul_helper(a, &unsqueezed_b); // (.... N, 1) 
            v.squeeze_view(v.shape.len()-1) // (..., M)
        }
        (1, _)=>{
            let unsqueezed_a = a.unsqueeze_view(0); // (1, M)
                                                            //b =>(..., M, N)
            let v = tensor_mul_helper(&unsqueezed_a, b); // (...., 1, N)
            v.squeeze_view(v.shape.len()-2) // (..., N)
            
        }
        (_, _)=>tensor_mul_helper(a, b)
    }
    
}


// (5) @ (5, 2) => (2)


pub fn matmul(tr: &mut Trace, a_id: NodeId, b_id: NodeId) -> NodeId{
    let  a= tr.get_tensor(a_id).clone();
    let  b = tr.get_tensor(b_id).clone();

    let a_rank = a.shape.len();
    let b_rank = b.shape.len();

    let c = tensor_mul(&a, &b);// moyen écrit comme ca. TODO: clean ce truc
    
    let vjp = move |g_out: &Tensor| -> SmallVec<[(NodeId, Tensor); 2]>{


        if a_rank == 1 && b_rank >=2{   
            let ga = tensor_mul(&g_out, &b.mat_transpose()).sum_over_broadcasted_batches(&a.shape);

            let a_col = a.unsqueeze_view(1); // ona du (M, 1)
            // (M, 1) @ (..., 1 N) => il faut ajouter  1 a gout en avant dernier
            let g_out_fixed = g_out.unsqueeze_view(g_out.shape.len()-1); 

            let gb = tensor_mul(&a_col, &g_out_fixed).sum_over_broadcasted_batches(&b.shape);
            smallvec![(a_id, ga), (b_id, gb)]

        }else if a_rank >=2 && b_rank == 1{
            let gb = tensor_mul(&a.mat_transpose(), &g_out).sum_over_broadcasted_batches(&b.shape);

            let b_lin = b.unsqueeze_view(0); // on a du (1, M)
            let g_out_fixed = g_out.unsqueeze_view(g_out.shape.len());

            let ga = tensor_mul(&g_out_fixed, &b_lin).sum_over_broadcasted_batches(&a.shape);

            smallvec![(a_id, ga), (b_id, gb)]

        }else{
            let ga = tensor_mul(&g_out, &b.mat_transpose()).sum_over_broadcasted_batches(&a.shape);
            let gb = tensor_mul(&a.mat_transpose(), &g_out).sum_over_broadcasted_batches(&b.shape);
            
            smallvec![(a_id, ga), (b_id, gb)]
        }

    }; 

    tr.push(crate::trace::Node { value: c, parents_id: smallvec![a_id, b_id], vjp: Some(Box::new(vjp)), is_param: false })
}