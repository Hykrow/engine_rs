(# engine_rs)

Petit moteur de calcul différentiel écrit en Rust — expérimental et pédagogique.

## But

Ce dépôt contient une implémentation minimale d'un graphe de calcul (tensors, opérations, gradients) destinée à expérimenter la construction d'opérations et la backpropagation.

## Comment builder

Ce projet utilise Cargo (Rust). Depuis la racine du dépôt :

```powershell
cargo build
cargo run
```

## Structure des fichiers

- `src/main.rs` : point d'entrée (exemple minimal).
- `src/tensor.rs` : définition de `Tensor` et logique liée aux tenseurs.
- `src/ops.rs` : définitions d'opérations (Add, Mul, Tanh).
- `src/graph.rs` : construction et gestion du graphe de calcul.

## Notes

- Projet en cours de développement.

N'hésitez pas si vous avez des corrections !

## Références et détails d'implémentation

# Calcul de gradients, Tenseurs
- Pour le calcul des gradients, j'ai énormément apprécié le blog de Robot Chinwag : https://robotchinwag.com/posts/the-tensor-calculus-you-need-for-deep-learning/ ceci m'a permis de programmer les différentes opérations sur les Tenseurs.  
 
- Pour comprendre le broadcast, https://numpy.org/doc/stable/user/basics.broadcasting.html est une bonne ressource.

# Moteur en lui-même. 
- Pour comprendre comment le moteur fonctionne dans son essence, la vidéo d'Andrej Kaparthy est très facile d'accès et complète: "RAJOUTER LIEN"
- Note: j'aimerais bien implémenter une version JIT si j'ai le temps...