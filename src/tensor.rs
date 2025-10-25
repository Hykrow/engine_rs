use std::ops::{Add, Mul};
use std::sync::Arc;
use std::iter; 
#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Arc<Vec<f32>>,
    pub shape : Vec<usize>, 
    pub strides : Vec<usize>, 
    pub offset : usize,
}
// TODO: remove les new innecessaires car problemes de recalcul de strides . eviter de recalculer les strides sauf a la vraie creation du tenseur initial.
// TODO: ajouter un DeviceType et un DataType

impl Tensor{

    // initialisations
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




    // changements de vues, transformations

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
    pub fn mat_transpose(&self) -> Tensor{
        let dim = self.shape.len();
        if dim == 1{
            self.clone()
        }else{
            let mut new_shape = self.shape.clone(); 
            let mut new_strides = self.strides.clone();
            new_shape.swap(dim-1, dim-2);
            new_strides.swap(dim-1, dim-2);
            Tensor { data: self.data.clone(), shape: new_shape.clone(), strides: new_strides, offset: self.offset }
        }

    }



    // conversion index, prise d'éléments, ... /!\ remove set2, inutile je pense..
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
    pub fn get_from_lin(&self, lin: usize) -> f32{
        let idxs = Tensor::idx_from_lin(&self.shape, lin);
        self.get(&idxs)
    }
    pub fn idx_from_lin(shape: &[usize], mut lin: usize) -> Vec<usize>{

        let mut idxs: Vec<usize> = shape.iter().rev().map(|&sa|
            if sa!=0 {
                let idx = lin%sa;
                lin/=sa;
                idx
            }else{
                 0
            }
        ).collect();
  
        // attention a ca ! +
        idxs.reverse();
        idxs
    }
    pub fn lin_from_idx(&self, id: &[usize])-> usize{
        let idx = id.iter().zip(self.strides.iter()).map(|(&a, &b)| a*b).sum::<usize>();
        idx
    }
    pub fn batch_offset(&self, idx: &[usize]) -> usize{
        idx.iter().zip(self.strides.iter()).map(|(&sh, &st)| sh*st).sum()
    }


    pub fn squeeze_first(&self, val:usize)->Tensor{
        let mut new_strides = self.strides.clone();
        let mut new_shape = self.shape.clone();
        for _ in 0..val{
            new_strides.remove(0);
            new_shape.remove(0);
        }
        Tensor{data:self.data.clone(), shape:new_shape, strides:new_strides, offset:self.offset}
    }

    /*

    Yb = A_b B  <= b un batch. Pour un tenseur, le nombre de batches c'est la somme des nouvelles shapes ayant été redimensionnées (broadcastées) (uniquement pour lobjet qu(on manipule))

           v--chain rule + def gradiant
    dL = sum <Gb, dYb> = sum <Gb dAb B + A_b dB>
       = sum <Gb, dAb B > + <Gb, A_b dB>
       = sum <Gb  B ^T , dAb> + <A_b ^T Gb, dB>

            ^GRADIANT Ab           ^ GRADIANT dB
        

     */ 
    pub fn sum_over_broadcasted_batches(&self, origin_shape : &[usize]) -> Tensor{




        println!("Tenseur d'origine : {}, shape à rematch : {:?}", self, origin_shape);
        let len_diff = self.shape.len()- origin_shape.len();
        let origin_shape_upd = [vec![1; len_diff], origin_shape.to_vec()].concat();
        let n = origin_shape_upd.len();
        let mut to_sum = vec![false; n];
        
        let mut new_shape = Vec::new();
        for i in 0..n{
            if origin_shape_upd[i] == 1 { 
                to_sum[i] = true;
                
                new_shape.push(1);// rechopper 1. 
            }else{
                new_shape.push(origin_shape_upd[i]);
            }
        }
        
        // calculer la nouvelle shape

        // mtn, on a les axes sur lesquels sommer.
        
        // il faut ensuite iterer sur chaque indice. quand on a un indice, on check lesquels fais : 
        /*
        faire une nouvelle shape avec les truc en moins -> fait
        nommons old_idx obtenu avec notre boucle for
        pour chauque dimension i : 
            ->si to_sum[i] vaut true, retirer ca de l'idx
        convertir le nouveaux idx en lin en passant en argument la nouvelle shape
        rajouter à vec[lin] self.value(old_idx)

         */
        //TODO: ajouter un truc pour check si ca a du sens  de faire ca dessus
        let mut new_data= vec![0f32; new_shape.numel()]; 
        let new_strides = Tensor::compute_strides(&new_shape);
        for lin in 0..(self.shape.numel()){
            let old_idx = Tensor::idx_from_lin(&self.shape, lin);
            let mut new_idx = Vec::new();
            for i in 0..n{
                if to_sum[i] == true{
                    new_idx.push(0);
                }else{
                    new_idx.push(old_idx[i]);
                }
            }
            let new_lin : usize= new_idx.iter().zip(new_strides.iter()).map(|(&sa, &st)| sa*st).sum();
            new_data[new_lin]+= self.get(&old_idx); // en vrai, vrm pas sur de ca. jsp si on a besoin de old idx et de new idx; et enocre moin de to_sum.. car je pens eque tt se fait automatiquement aec les strides

        }

        let t = Tensor{
            data:Arc::new(new_data), 
            shape: new_shape.clone(), 
            strides: Tensor::compute_strides(&new_shape), 
            offset: 0
        }.squeeze_first(len_diff); 
        println!("Tenseur transformé et sommé : {}", t);
        t

    }



    // fonction generique
    pub fn apply<F>(&self, f : F)-> Tensor
    where F : Fn(f32) -> f32
    {
        
        let new_values = Arc::new(self.data.iter().map(|&x| f(x)).collect());
        Tensor { data: new_values, shape: self.shape.clone(), strides: self.strides.clone(), offset: self.offset }
        
    }
}

pub trait Numel {
    fn numel(&self) -> usize;
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

        // Gestion des tenseurs vides (au moins une dim = 0)
        if self.shape.iter().any(|&d| d == 0) {
            writeln!(f, "  data: []")?;
            return write!(f, ")");
        }

        let rank = self.shape.len();

        // Scalar (0D)
        if rank == 0 {
            // offset pointe sur l'élément
            let v = self.data[self.offset];
            writeln!(f, "  data: {:.4}", v)?;
            return write!(f, ")");
        }

        // Petite fonction d'indentation
        #[inline]
        fn indent(f: &mut fmt::Formatter<'_>, n: usize) -> fmt::Result {
            for _ in 0..n { write!(f, " ")?; }
            Ok(())
        }

        // Impression récursive d'un N-D array en respectant strides/offset.
        fn fmt_rec(
            t: &Tensor,
            f: &mut fmt::Formatter<'_>,
            dim: usize,
            idx: &mut [usize],
            base_indent: usize, // indentation de départ (par ex. 2)
        ) -> fmt::Result {
            let rank = t.shape.len();

            if dim + 1 == rank {
                // Dernière dimension : on imprime une "ligne" 1D alignée.
                write!(f, "[")?;
                for i in 0..t.shape[dim] {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    idx[dim] = i;
                    let v = t.get(idx);
                    write!(f, "{:8.4}", v)?;
                }
                write!(f, "]")?;
                Ok(())
            } else {
                // Dimensions supérieures : on ouvre un bloc, puis chaque sous-tranche sur une nouvelle ligne.
                write!(f, "[")?;
                for i in 0..t.shape[dim] {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    writeln!(f)?;
                    indent(f, base_indent + 2)?;
                    idx[dim] = i;
                    fmt_rec(t, f, dim + 1, idx, base_indent + 2)?;
                }
                writeln!(f)?;
                indent(f, base_indent)?;
                write!(f, "]")?;
                Ok(())
            }
        }

        // Cas 1D/2D bénéficient du même moteur mais on garde l’en-tête "data:" + jolie mise en page.
        write!(f, "  data: ")?;
        let mut idx = vec![0usize; rank];
        if rank == 1 {
            // Simple ligne
            fmt_rec(self, f, 0, &mut idx, 2)?;
            writeln!(f)?;
        } else if rank == 2 {
            // Matrice : chaque ligne sur sa propre ligne déjà géré par fmt_rec
            fmt_rec(self, f, 0, &mut idx, 2)?;
            writeln!(f)?;
        } else {
            // N-D : idem, avec indentations
            fmt_rec(self, f, 0, &mut idx, 2)?;
            writeln!(f)?;
        }

        write!(f, ")")
    }
}