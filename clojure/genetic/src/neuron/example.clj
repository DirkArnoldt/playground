(ns neuron.example
  (:require [neuron.core :as n])
  (:require [neuron.hopfield :as h])
  (:require [neuron.mlp :as mlp])
  (:require [clojure.java.io :as io]))

(def digits [
             [-1.0 -1.0 -1.0 -1.0 -1.0
              -1.0  1.0  1.0  1.0 -1.0
              -1.0  1.0 -1.0  1.0 -1.0
              -1.0  1.0 -1.0  1.0 -1.0
              -1.0  1.0 -1.0  1.0 -1.0
              -1.0  1.0  1.0  1.0 -1.0
              -1.0 -1.0 -1.0 -1.0 -1.0]
             [-1.0 -1.0 -1.0 -1.0 -1.0
              -1.0 -1.0 -1.0  1.0 -1.0
              -1.0 -1.0 -1.0  1.0 -1.0
              -1.0 -1.0 -1.0  1.0 -1.0
              -1.0 -1.0 -1.0  1.0 -1.0
              -1.0 -1.0 -1.0  1.0 -1.0
              -1.0 -1.0 -1.0 -1.0 -1.0]
             [-1.0 -1.0 -1.0 -1.0 -1.0
              -1.0  1.0  1.0  1.0 -1.0
              -1.0 -1.0 -1.0  1.0 -1.0
              -1.0  1.0  1.0  1.0 -1.0
              -1.0  1.0 -1.0 -1.0 -1.0
              -1.0  1.0  1.0  1.0 -1.0
              -1.0 -1.0 -1.0 -1.0 -1.0]
             [-1.0 -1.0 -1.0 -1.0 -1.0
              -1.0  1.0  1.0  1.0 -1.0
              -1.0 -1.0 -1.0  1.0 -1.0
              -1.0  1.0  1.0  1.0 -1.0
              -1.0 -1.0 -1.0  1.0 -1.0
              -1.0  1.0  1.0  1.0 -1.0
              -1.0 -1.0 -1.0 -1.0 -1.0]
             [-1.0 -1.0 -1.0 -1.0 -1.0
              -1.0  1.0 -1.0  1.0 -1.0
              -1.0  1.0 -1.0  1.0 -1.0
              -1.0  1.0  1.0  1.0 -1.0
              -1.0 -1.0 -1.0  1.0 -1.0
              -1.0 -1.0 -1.0  1.0 -1.0
              -1.0 -1.0 -1.0 -1.0 -1.0]
             [-1.0 -1.0 -1.0 -1.0 -1.0
              -1.0  1.0  1.0  1.0 -1.0
              -1.0  1.0 -1.0 -1.0 -1.0
              -1.0  1.0  1.0  1.0 -1.0
              -1.0 -1.0 -1.0  1.0 -1.0
              -1.0  1.0  1.0  1.0 -1.0
              -1.0 -1.0 -1.0 -1.0 -1.0]
             [-1.0 -1.0 -1.0 -1.0 -1.0
              -1.0  1.0  1.0  1.0 -1.0
              -1.0  1.0 -1.0 -1.0 -1.0
              -1.0  1.0  1.0  1.0 -1.0
              -1.0  1.0 -1.0  1.0 -1.0
              -1.0  1.0  1.0  1.0 -1.0
              -1.0 -1.0 -1.0 -1.0 -1.0]
             [-1.0 -1.0 -1.0 -1.0 -1.0
              -1.0  1.0  1.0  1.0 -1.0
              -1.0 -1.0 -1.0  1.0 -1.0
              -1.0 -1.0 -1.0  1.0 -1.0
              -1.0 -1.0 -1.0  1.0 -1.0
              -1.0 -1.0 -1.0  1.0 -1.0
              -1.0 -1.0 -1.0 -1.0 -1.0]
             [-1.0 -1.0 -1.0 -1.0 -1.0
              -1.0  1.0  1.0  1.0 -1.0
              -1.0  1.0 -1.0  1.0 -1.0
              -1.0  1.0  1.0  1.0 -1.0
              -1.0  1.0 -1.0  1.0 -1.0
              -1.0  1.0  1.0  1.0 -1.0
              -1.0 -1.0 -1.0 -1.0 -1.0]
             [-1.0 -1.0 -1.0 -1.0 -1.0
              -1.0  1.0  1.0  1.0 -1.0
              -1.0  1.0 -1.0  1.0 -1.0
              -1.0  1.0  1.0  1.0 -1.0
              -1.0 -1.0 -1.0  1.0 -1.0
              -1.0  1.0  1.0  1.0 -1.0
              -1.0 -1.0 -1.0 -1.0 -1.0]
             ])

(defn -main-h [& args]
  (time
    (let [n (h/build 35 :hebbian)
          n' (n/train n digits)]
      (doall
        (for [i (range 10)]
          (do
          (println (digits i))
          (println (n/recall n' (digits i)))
          (println "---")))))))

(defn read-lines [filename]
  (with-open [rdr (io/reader filename)]
    (doall (line-seq rdr))))

(defn as-vector [s delim]
  (vec (.split s delim)))

(defn to-doubles [coll]
  (vec (map #(Double/parseDouble %) coll)))

(defn split-input-output [line]
  (let [[in out] (as-vector line ":")
        input (to-doubles (as-vector in " "))
        output (to-doubles (as-vector out " "))]
    (vector input output)))

(defn make-training-set [lines]
  (map #(split-input-output %) lines))


(defn -main [& args]
  (time
    (let [lines (read-lines "resources/mlp-training.dat")
          training-set (make-training-set lines)
          sample (first training-set)
          inodes (count (first sample))
          onodes (count (second sample))
          n (mlp/build [inodes inodes onodes] {})
          trained (n/train n (apply concat (repeat 100 training-set)))
          output (n/recall trained (first sample))
          error (mlp/error (second sample) output)]
      (do
        (println "sample")
        (println sample)
        (println "output")
        (println output)
        (println "error")
        (println error)
        (println "---")))))