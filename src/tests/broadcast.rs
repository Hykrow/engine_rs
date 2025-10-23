//ecrit avec ChatGPT

use crate::tensor::Tensor;
use std::sync::Arc;
#[cfg(test)]

fn contig_strides(shape: &[usize]) -> Vec<usize> {
    // Strides row-major (C) : produit suffixe
    let mut strides = vec![0; shape.len()];
    let mut acc = 1;
    for (i, &dim) in shape.iter().enumerate().rev() {
        strides[i] = acc;
        acc *= dim.max(1);
    }
    strides
}

fn mk_tensor(shape: &[usize]) -> Tensor {
    let strides = contig_strides(shape);
    let size: usize = shape.iter().product::<usize>().max(1);
    Tensor {
        data: Arc::new(vec![0.0; size]), // dummy
        shape: shape.to_vec(),
        strides,
        offset: 0,
    }
}

#[test]
fn broadcast_same_shape_keeps_strides() {
    let x = mk_tensor(&[2, 3]);
    let y = x.broadcast_view(&[2, 3]).unwrap();
    assert_eq!(y.shape, vec![2, 3]);
    assert_eq!(y.strides, x.strides);
}

#[test]
fn broadcast_expand_from_1_to_n_sets_stride_zero() {
    // [1,3] -> [2,3]
    let x = mk_tensor(&[1, 3]);
    let y = x.broadcast_view(&[2, 3]).unwrap();
    assert_eq!(y.shape, vec![2, 3]);
    // axe 0 était 1 -> 2 : stride doit être 0
    assert_eq!(y.strides[0], 0);
    // axe 1 identique : stride conservé
    assert_eq!(y.strides[1], x.strides[1]);
}

#[test]
fn broadcast_left_pad_and_expand_multiple_axes() {
    // source [3,1] → cible [2,3,4]
    // left-pad: [1,3,1] puis broadcast -> [2,3,4]
    let x = mk_tensor(&[3, 1]);
    let y = x.broadcast_view(&[2, 3, 4]).unwrap();
    assert_eq!(y.shape, vec![2, 3, 4]);

    // axe 0 : 1 -> 2 => stride 0
    assert_eq!(y.strides[0], 0);
    // axe 1 : 3 -> 3 => stride conservé (c'est l'ancien axe 0 de x après unsqueeze)
    // On ne peut pas affirmer l'égalité exacte sans connaître unsqueeze_first,
    // mais au minimum, il ne doit PAS être 0.
    assert_ne!(y.strides[1], 0);
    // axe 2 : 1 -> 4 => stride 0
    assert_eq!(y.strides[2], 0);
}

#[test]
fn broadcast_all_ones_to_any_all_strides_zero() {
    let x = mk_tensor(&[1, 1, 1]);
    let y = x.broadcast_view(&[5, 6, 7]).unwrap();
    assert_eq!(y.shape, vec![5, 6, 7]);
    assert_eq!(y.strides, vec![0, 0, 0]);
}

#[test]
fn error_when_shrinking_dim_s_gt_1_to_t_eq_1() {
    // ⚠️ `broadcast_view(self, target)` NE DOIT PAS autoriser s>1 -> t==1
    let x = mk_tensor(&[2, 3]);
    let err = x.broadcast_view(&[1, 3]).unwrap_err();
    assert!(
        err.contains("non broadcastable") || err.contains("incompat"),
        "err was: {err}"
    );
}

#[test]
fn error_when_target_dim_incompatible_and_not_one() {
    let x = mk_tensor(&[2, 3]);
    let err = x.broadcast_view(&[4, 3]).unwrap_err();
    assert!(
        err.contains("non broadcastable") || err.contains("incompat"),
        "err was: {err}"
    );
}

#[test]
fn error_when_target_rank_too_small() {
    let x = mk_tensor(&[2, 3, 4]);
    let err = x.broadcast_view(&[3, 4]).unwrap_err();
    assert!(
        err.contains("trop petite") || err.contains("rank"),
        "err was: {err}"
    );
}

#[test]
fn ok_when_both_are_one_dim_stays_one_stride_kept() {
    let x = mk_tensor(&[1, 5, 1]);
    let y = x.broadcast_view(&[1, 5, 1]).unwrap();
    assert_eq!(y.shape, vec![1, 5, 1]);
    assert_eq!(y.strides, x.strides);
}

