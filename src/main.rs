use lamp::trace::Trace;
use lamp::utils;
use lamp::utils::params::get_params_id;
use mnist::MnistBuilder;
use lamp::autodiff::value_and_grad::value_and_grad;
use lamp::dataloader::dataloader::DataLoader;
use lamp::data_examples::mnist_data::MnistDataset;
use lamp::data_examples::mnist_data::collate_mnist_xy_u8_to_tensors;
use lamp::nn::functions::relu;
use lamp::nn::layers::bind::ParamCursor;
use lamp::nn::layers::linear::Linear;
use lamp::nn::losses::softmax_crossentropy;
use lamp::optim::sgd;

fn main() {
    let rows = 28; let cols = 28;
    let mn = MnistBuilder::new()
        .base_path("data")
        .label_format_digit()
        .training_set_length(6000)
        .test_set_length(500)
        .finalize();

    let ds_train = MnistDataset { imgs: mn.trn_img, labs: mn.trn_lbl, rows, cols };
    let ds_test  = MnistDataset { imgs: mn.tst_img, labs: mn.tst_lbl, rows, cols };

    //TODO: check ce collate, et check le collate example; minst. 
    // Collate partiellement appliquée (rows/cols/k/flatten fixés)
    let collate_train = |batch: Vec<(Vec<u8>, u8)>| collate_mnist_xy_u8_to_tensors(batch, rows, cols, 10, true);
    let collate_test  = |batch: Vec<(Vec<u8>, u8)>| collate_mnist_xy_u8_to_tensors(batch, rows, cols, 10, true);

    let mut train = DataLoader::new(ds_train, 10_000, true,  collate_train);
    let mut test  = DataLoader::new(ds_test,  2_000, false, collate_test);



    let mut params = vec![ // TODO: retirer le concat et prendre un Vec<Vec<Tensor>>, un pour chaque layer ? 
        Linear::init_kaiming(784, 200),
        Linear::init_kaiming(200, 50), 
        Linear::init_kaiming(50, 10)
    ].concat();
    let sgd = sgd::Sgd {lr: 0.1};

    //TODO: ré écrire ICI..
    fn forward_logits(tr: &mut Trace, pids: &[usize], x: usize) -> usize {
        let mut cur = ParamCursor::new(pids);
        // IL EST TRES IMPORTANT DE BIND DANS LE MEME ORDRE QUE PREVU 
        let l1 = Linear::bind(&mut cur);
        let l2 = Linear::bind(&mut cur);
        let l3 = Linear::bind(&mut cur);

        let h1 = l1.apply(tr, x);
        let z1 = relu(tr, h1);
        let h2 = l2.apply(tr, z1);
        let z2 = relu(tr, h2);
        l3.apply(tr, z2) // logits
    }

    // TODO: faire une fonction model qui prend x, y, tr, pids et qui renvoie la fonction build tr pid pour pas avoir a copier coller a l'inférence aussi
    for epoch in 0..10 {
        train.reset_epoch();
        for (xb, yb) in &mut train {
            let (loss, grads) = value_and_grad(&params, |tr, pids| {
                let x = tr.input(xb.clone());
                let y = tr.input(yb.clone());
                let logits = forward_logits(tr, pids, x);
                softmax_crossentropy(tr, logits, y) 
            });

        println!("loss: {}", loss.data[0]);
        params = sgd.update(&params, &grads); 
        }
        println!("epoch {epoch} ok");
    }

    // maintenant, passons à l'inférence: 
    let mut correct = 0usize;
    let mut total = 0usize;
    test.reset_epoch();

    //TODO: utiliser une fonction inférence à la place..
    for (xb, yb) in &mut test {
        let mut tr = Trace::new();
        
        let x = tr.input(xb.clone());
        let pids = get_params_id(&mut tr,&params); 
        let logits = forward_logits(&mut tr, &pids, x);
        let pred = tr.get_tensor(logits).argmax_last();      // [B]
        let y_true = yb.argmax_last(); // si one-hot
        
        correct += pred.iter().zip(y_true.iter()).map(|(&a, &b)| if a == b {1} else {0}).sum::<usize>(); // compte les égaux dans le batch
        total += xb.shape[0]; // ici ok car 1d de batch et 1d de vecteur
    }
    println!("accuracy = {:.2}%", 100.0 * correct as f32 / total as f32);

}
