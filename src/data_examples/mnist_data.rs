use crate::dataloader::dataloader::Dataset;
use crate::tensor::Tensor;

pub struct MnistDataset {
    pub imgs: Vec<u8>,  // N * 28 * 28
    pub labs: Vec<u8>,  // N
    pub rows: usize,
    pub cols: usize,
}

impl Dataset for MnistDataset {
    type Item = (Vec<u8>, u8); // item brut, pas d'hypothèse Tensor ici

    fn len(&self) -> usize { self.labs.len() }

    fn get(&self, index: usize) -> Self::Item {
        let img_sz = self.rows * self.cols;
        let off = index * img_sz;
        let img = self.imgs[off..off+img_sz].to_vec(); // copie: OK pour MNIST
        let y = self.labs[index];
        (img, y)
    }
}

pub fn collate_mnist_xy_u8_to_tensors(items: Vec<(Vec<u8>, u8)>, rows: usize, cols: usize, num_classes: usize, flatten: bool) -> (Tensor, Tensor) {
    let b = items.len();
    let img_sz = rows * cols;

    // X
    let mut x = Vec::with_capacity(b * img_sz);
    for (img, _) in &items {
        assert_eq!(img.len(), img_sz);
        for px in img { x.push(*px as f32 / 255.0); }
    }
    let x_shape = if flatten { vec![b, img_sz] } else { vec![b, 1, rows, cols] }; // pour cnns
    let xb = Tensor::from_vec(&x, &x_shape).unwrap();

    // Y one-hot
    let mut y = vec![0.0f32; b * num_classes];
    for (i, (_, lab)) in items.iter().enumerate() {
        y[i * num_classes + (*lab as usize)] = 1.0;
    }
    let yb = Tensor::from_vec(&y, &[b, num_classes]).unwrap();
    (xb, yb)
}
