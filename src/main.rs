use crate::tensor::Tensor;

mod tensor;
mod graph; 

mod ops;

use ops::Op;


fn t_seq(shape: &[usize]) -> Tensor {
    let size: usize = shape.iter().product();
    let data: Vec<f32> = (1..=size).map(|x| x as f32).collect();
    Tensor::from_vec(&data, shape).unwrap()
}

fn main() {
    // --- Test A : parent [1,3] broadcasté en [2,3]
    {
        let origin = [1, 3];
        let x = Tensor::from_vec(&[1.0, 2.0, 3.0], &origin).unwrap();
        let x_b = x.broadcast_view(&[2, 3]).unwrap(); // vue broadcastée
        let red = x_b.sum_over_broadcasted_batches(&origin);
        println!("=== Test A ===");
        println!("origin shape: {:?}\nBroadcast view:", origin);
        println!("{x_b}");
        println!("Reduced back to parent:");
        println!("{red}");
    }

    // --- Test B : parent [2,1,4] broadcasté en [2,3,4]
    {
        let origin = [2, 1, 4];
        let a = t_seq(&origin);
        let a_b = a.broadcast_view(&[2, 3, 4]).unwrap();
        let red = a_b.sum_over_broadcasted_batches(&origin);
        println!("=== Test B ===");
        println!("origin shape: {:?}\nBroadcast view:", origin);
        println!("{a_b}");
        println!("Reduced back to parent:");
        println!("{red}");
    }

    // --- Test C : parent [1,1] (scalaire 2D) broadcasté en [2,3]
    {
        let origin = [1, 1];
        let s = Tensor::from_vec(&[5.0], &origin).unwrap();
        let s_b = s.broadcast_view(&[2, 3]).unwrap();
        let red = s_b.sum_over_broadcasted_batches(&origin);
        println!("=== Test C ===");
        println!("origin shape: {:?}\nBroadcast view:", origin);
        println!("{s_b}");
        println!("Reduced back to parent:");
        println!("{red}");
    }

    // --- Test D : aucun axe à réduire (parent == self)
    {
        let origin = [2, 3];
        let m = t_seq(&origin);
        let red = m.sum_over_broadcasted_batches(&origin);
        println!("=== Test D ===");
        println!("origin/self shape: {:?}\nSelf tensor:", origin);
        println!("{m}");
        println!("Reduced back to parent (should keep same shape):");
        println!("{red}");
    }

    // --- Test E : parent [1,4] broadcasté en [2,3,4] (padding à gauche)
    {
        let origin = [1, 4];
        let v = t_seq(&origin);
        let v_b = v.broadcast_view(&[2, 3, 4]).unwrap();
        let red = v_b.sum_over_broadcasted_batches(&origin);
        println!("=== Test E ===");
        println!("origin shape: {:?}\nBroadcast view:", origin);
        println!("{v_b}");
        println!("Reduced back to parent:");
        println!("{red}");
    }
}
