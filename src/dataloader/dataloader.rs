use rand::{seq::SliceRandom, thread_rng};

pub trait Dataset{
    type Item: Clone;
    fn len(&self) -> usize;
    fn get(&self, idx: usize) -> Self::Item;  
}
pub struct DataLoader<D, C, B>
where 
    D: Dataset, 
    C: Fn(Vec<D::Item>) -> B
    
{
    dataset: D,  
    indices: Vec<usize>, 
    bs: usize, 
    pos: usize, 
    shuffle: bool, 
    collate: C

}

impl <D, C, B> DataLoader<D, C, B>
where 
    D: Dataset, 
    C: Fn(Vec<D::Item>) -> B
{
    pub fn new(dataset: D, batch_size: usize, shuffle: bool, collate: C) -> Self{
        let mut indices: Vec<_> = (0..dataset.len()).collect();
        if shuffle{
            indices.shuffle(&mut thread_rng());
        }
        Self { dataset, indices, bs: batch_size, pos: 0, shuffle, collate }
    }

    pub fn reset_epoch(&mut self){
        self.pos = 0; 
        if self.shuffle{
            self.indices.shuffle(&mut thread_rng());
        }
    }
}


impl <D, C, B> Iterator for DataLoader<D, C, B>
where
    D: Dataset, 
    C: Fn(Vec<D::Item>) -> B
{
    type Item = B;

    #[inline]
    fn next(&mut self) -> Option<Self::Item>{
        if self.pos >=self.indices.len(){
            return None; 
        }
        let to = (self.pos+self.bs).min(self.indices.len());

        let idxs = &self.indices[self.pos..to];     
        self.pos= to; 
        
        let mut next_batch =  Vec::new();

        for &i in idxs{
            next_batch.push(self.dataset.get(i));
        }
        Some((self.collate)(next_batch))
    }

    #[inline]
    fn size_hint(& self) -> (usize, Option<usize>) {
        let remaining_items = self.indices.len().saturating_sub(self.pos);
        let batches = (remaining_items +self.bs -1)/self.bs; // ceil 
        (batches, Some(batches))
    }
}