use std::ops::{Add, Mul};
use std::sync::Arc;

#[derive(Debug)]
pub struct Tensor {
    pub data: Arc<Vec<f32>>,
    pub shape : Vec<usize>, 
    pub strides : Vec<usize>, 
    pub offset : usize,
}


impl Tensor{

    pub fn new(data: Arc<Vec<f32>>, shape :&[usize], offset: usize) -> Tensor{
        Tensor{
            data, 
            shape : shape.to_vec(),
            strides: Tensor::compute_strides(shape), 
            offset
        }
    }
    pub fn compute_strides(shape : &[usize]) -> Vec<usize>{
        let n = shape.len();
        let mut strides = vec![0; n];
        let mut product = 1; 
        for (i, dim) in shape.iter().rev().enumerate(){
            strides[n-i-1] = product; 
            product*=dim;
        }
        strides
    }
    pub fn ones(shape : &[usize])-> Tensor{
        Tensor{
            data: Arc::new(vec![1.0; shape.iter().product()]), 
            shape : shape.to_vec(), 
            strides : Self::compute_strides(shape), 
            offset: 0
        }
    } 
    pub fn zeros(shape : &[usize])-> Tensor{
        Tensor{
            data: Arc::new(vec![0.0; shape.iter().product()]), 
            shape : shape.to_vec(), 
            strides : Self::compute_strides(shape), 
            offset: 0
        }
    } 
    pub fn from_vec(data : &[f32], shape : &[usize]) -> Result<Tensor, String>{
        if data.len() != shape.iter().product(){
            return Err("taille data non conforme a la shape".into());
        }
        Ok(
            Tensor { data: Arc::new(data.to_vec()), shape: shape.to_vec(), strides: Tensor::compute_strides(shape), offset:0
        })
    }
    pub fn from_owned(shape: &[usize], data: Vec<f32>) -> Result<Self, String> {
        if data.len() != shape.iter().product(){
            return Err("taille data non conforme a la shape".into());
        }
        Ok(Tensor {
            data: Arc::new(data),
            shape: shape.to_vec(),
            strides: Tensor::compute_strides(shape),
            offset: 0,
        })
    }

    #[inline(always)]
    pub fn is_broadcasted(&self) -> bool {
        self.strides.iter().any(|&s| s == 0)
    }

    pub fn unsqueeze_view(&self, axis: usize) -> Tensor{
        let mut new_shape = self.shape.clone(); 
        new_shape.insert(axis, 1);
        Tensor::new(self.data.clone(), &new_shape, self.offset)
    }
    pub fn squeeze_view(& self, axis: usize) ->Tensor{
        let mut new_shape = self.shape.clone(); 
        new_shape.remove(axis);

        Tensor::new(self.data.clone(), &new_shape, self.offset)
    }
    pub fn broadcast_view() -> Tensor{
        
    }

    #[inline(always)] // pour la rapitidité
    pub fn get2(&self, i: usize, j: usize) -> f32 {
        self.data[self.offset + self.strides[0]*i + self.strides[1]*j]
    }
    #[inline(always)] // pour la rapidité
    pub fn set2(&mut self, i: usize, j: usize, v: f32) {
        assert!(!self.is_broadcasted(), "impossible de set une value sur des tenseurs broadcastés");  // TODO : changer par une variable bool plutot qu'un appel à une fonction à chaque fois. mais faire attention à bien upload à chaque broadcast bien ... 
        let data: &mut Vec<f32> = Arc::get_mut(&mut self.data).expect("la ref est partagé, impossible de modifier"); 
        data[self.offset+ self.strides[0]*i + self.strides[1]*j] = v;
    }
    pub fn get_value(&self, id: &[usize]) -> f32{   
        let idx = id.iter().zip(self.strides.iter()).map(|(&a, &b)| a*b).sum::<usize>();
        self.data[idx+self.offset]
    }
}

