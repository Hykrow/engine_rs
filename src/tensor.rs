use std::ops::{Add, Mul};
use std::sync::Arc;
use std::iter; 
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
    pub fn from_owned(data: Vec<f32>, shape: &[usize]) -> Result<Self, String> {
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

    pub fn unsqueeze_view(&self, axis: usize) -> Tensor{// TODO : jouter l'axis negatif
        let mut new_shape: Vec<usize> = self.shape.clone(); 
        new_shape.insert(axis, 1);
        Tensor::new(self.data.clone(), &new_shape, self.offset)
    }
    pub fn squeeze_view(& self, axis: usize) ->Tensor{ // TODO : jouter l'axis negatif
        let mut new_shape = self.shape.clone(); 
        new_shape.remove(axis);

        Tensor::new(self.data.clone(), &new_shape, self.offset)
    }
    pub fn unsqueeze_first(&self, nb_dim: usize) -> Tensor{// TODO : jouter l'axis negatif
        let new_shape = [vec![1; nb_dim], self.shape.clone()].concat();
        Tensor::new(self.data.clone(), &new_shape, self.offset)
    }

    pub fn broadcast_view(&self, a: &[usize]) -> Result<Tensor, String>{
        //left pad d'abord: 
        if a.len() < self.shape.len(){
            return Err("len de la shape de la cible trop petite".into());
        }
        let mut res: Tensor = self.unsqueeze_first(a.len()-self.shape.len()); // assumes that la shape quon veut a une taille >= celle quon aura
        assert_eq!(res.shape.len(), a.len(), "enorme bug broadcast_view");
        //recalculer les strides
        let n  = a.len();
        for i in 0..n{
            if res.shape[i] < a[i] && res.shape[i] ==1 {
                res.shape[i] =  a[i];
                res.strides[i] = 0;
            }else if res.shape[i] != a[i] && res.shape[i] != 1 { // pas la meme shape mais pas broadcastable...
                return Err("broadcast_view : non broadcastable..".into());
            }
        }
       Ok(res)
        
    }

    // gives the needed shape for the two tensors
    pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>, String>{
        let n = a.len().max(b.len());

        let mut ita = a.iter().rev().copied().chain(iter::repeat(1)); 
        let mut itb = b.iter().rev().copied().chain(iter::repeat(1));

        let mut res = Vec::with_capacity(n);

        for _ in 0..n{ // juste pour iterer n fois sur les iterateurs..
      
            let el_a = ita.next().unwrap();
            let el_b = itb.next().unwrap();
 
            if el_a == 1 || el_b == 1 || el_a == el_b {
                res.push(el_a.max(el_b));
            }else{
                return Err("Error broadcast_shape".into());
            }
        }
        res.reverse();
        Ok(res)

    }

    pub fn vue2d(&self, offset: usize) -> Tensor{
        let dim = self.shape.len();
        Tensor { data: self.data.clone(), shape: self.shape[(dim-2)..dim].to_vec(), strides: self.strides[(dim-2)..dim].to_vec(), offset: offset }
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
    #[inline(always)]
    pub fn get(&self, id: &[usize]) -> f32{   
        let idx = id.iter().zip(self.strides.iter()).map(|(&a, &b)| a*b).sum::<usize>();
        self.data[idx+self.offset]
    }
    #[inline(always)]
    pub fn get_from_lin(&self, mut lin: usize) -> f32{
        let mut idxs = Vec::with_capacity(self.shape.len());
        for i in (0..(idxs.len())).rev()
        {
            if(self.strides[i]!=0){
                idxs.push(lin%self.strides[i]);
                lin/=self.strides[i];
            }
        }
        self.get(&idxs)
    }
    pub fn idx_from_lin(shape: &[usize], mut lin: usize) -> Vec<usize>{
        let mut idxs = Vec::with_capacity(shape.len());
        for i in (0..(shape.len())).rev()
        {
            if(shape[i]!=0){
                idxs.push(lin%shape[i]);
                lin/=shape[i];
            }
        }
        // attention a ca ! +
        idxs.reverse();
        idxs
    }
    pub fn batch_offset(&self, idx: &[usize]) -> usize{
        idx.iter().zip(self.strides.iter()).map(|(&sh, &st)| sh*st).sum()
    }
}

pub trait Numel {
    fn numel(&self) -> usize;
}

impl Numel for Tensor {
    fn numel(&self) -> usize {
        self.shape.iter().zip(self.strides.iter()).map(|(&sa, &st)|
    if st == 0 {
        1
    }else{
       sa
    }).product() 
    }
}

impl Numel for [usize] {
    fn numel(&self) -> usize {
        self.iter().product()
    }
}

impl Numel for Vec<usize>{
    fn numel(&self) -> usize {
        self.iter().product()
    }
}



use std::fmt;

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Tensor(")?;
        writeln!(f, "  shape: {:?},", self.shape)?;
        writeln!(f, "  strides: {:?},", self.strides)?;
        writeln!(f, "  offset: {},", self.offset)?;

        match self.shape.len() {
            0 => writeln!(f, "  data: []")?,
            1 => {
                // vecteur
                let n = self.shape[0];
                let vals: Vec<String> = self.data[..n]
                    .iter()
                    .map(|x| format!("{:.4}", x))
                    .collect();
                writeln!(f, "  data: [{}]", vals.join(", "))?;
            }
            2 => {
                // matrice 2D
                let (rows, cols) = (self.shape[0], self.shape[1]);
                writeln!(f, "  data: [")?;
                for i in 0..rows {
                    let start = i * cols;
                    let end = start + cols;
                    let vals: Vec<String> = self.data[start..end]
                        .iter()
                        .map(|x| format!("{:8.4}", x))
                        .collect();
                    writeln!(f, "    [{}],", vals.join(", "))?;
                }
                writeln!(f, "  ]")?;
            }
            _ => {
                let preview = 10.min(self.data.len());
                writeln!(
                    f,
                    "  data: {:?} ... ({} éléments, dim>{})",
                    &self.data[..preview],
                    self.data.len(),
                    self.shape.len()
                )?;
            }
        }

        write!(f, ")")
    }
}
