use crate::dataloader::dataloader::Dataset;
use crate::tensor::Tensor;


// Ici, le dataloader a moyen sens car on charge tout direcement dans la mémoire. cependant, si jamais un a une fonction get-> qui peut se faire au fur et à mesure, ca serait beaucoup plus intéressant...

pub struct MnistDataset {
    pub imgs: Vec<u8>,  // N * 28 * 28
    pub labs: Vec<u8>,  // N
    pub rows: usize,
    pub cols: usize,
}

impl Dataset for MnistDataset { 
    type Item = (Vec<u8>, u8); 

    fn len(&self) -> usize { self.labs.len() }

    fn get(&mut self, index: usize) -> Self::Item { // comme écrit en haut, pas besoin de modifier car ok de tout charger. 
                                                    // sinon on aurait ajouté des truc genre no_idx dans MnistDataset afin de
                                                    //pouvoir le mettre à jour et ne load en mémoire que les no_idx + batch_size derniers éléments. 
                                                    // Mais, c'est assez bizarre si jamais on shuffle. 
        let img_sz = self.rows * self.cols;
        let off = index * img_sz;
        let img = self.imgs[off..off+img_sz].to_vec(); 
        let y = self.labs[index];
        (img, y)
    }
}

pub fn collate_mnist_xy_u8_to_tensors(items: Vec<(Vec<u8>, u8)>, rows: usize, cols: usize, num_classes: usize, flatten: bool) -> (Tensor, Tensor) {
    let b = items.len();
    let img_sz = rows * cols;

    //x
    let mut x = Vec::with_capacity(b * img_sz);
    for (img, _) in &items {
        assert_eq!(img.len(), img_sz);
        for px in img { x.push(*px as f32 / 255.0); }
    }
    let x_shape = if flatten { vec![b, img_sz] } else { vec![b, 1, rows, cols] }; // pour cnns
    let xb = Tensor::from_vec(&x, &x_shape).unwrap();

    //y 
    let mut y = vec![0.0f32; b * num_classes];
    for (i, (_, lab)) in items.iter().enumerate() {
        y[i*num_classes + (*lab as usize)]= 1f32;
    }
    let yb = Tensor::from_vec(&y, &[b, num_classes]).unwrap();
    (xb, yb)
}
