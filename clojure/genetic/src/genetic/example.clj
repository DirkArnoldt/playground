(ns genetic.example
  (:require genetic.core)
  (:import (genetic.core Chromosome)))

(defn make []
  (repeatedly 10 #(rand-int 256)))

(def target (make))

(defn calcfitness [v]
  (Math/abs (- 1.0 (/ (reduce + (mapv #(Math/abs (int (- %1 %2))) v target)) 2550))))

(defrecord NumChromosome [value fitness]
  Chromosome
  (mutate [_]
    (let [split (rand-int 9)
          v (concat (take split value) (vector (rand-int 256)) (drop (+ split 1) value))]
      (NumChromosome. v (calcfitness v))))
  (recombine [_ rhs]
    (let [split (rand-int 10)
          v (concat (take split value) (drop split (:value rhs)))]
      (NumChromosome. v (calcfitness v))))
  (calc-fitness [_]
    fitness))

(defn make-chromo []
  (let [v (make)]
    (NumChromosome. v (calcfitness v))))

(defn make-population
  ([n]
   (repeatedly n make-chromo))
  ([]
   (repeatedly 100 make-chromo)))
