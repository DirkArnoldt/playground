(ns neuron.example
  (:require [neuron.core :as n])
  (:require [neuron.hopfield :as h])
  (:require [neuron.mlp :as mlp])
  (:require [util.activation :as a])
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

(defn- test-net [net test-set]
  (let [total (count test-set)]
    (loop [acc 0
           errors 0.0
           test-set test-set]
      (if (seq test-set)
        (let [sample (first test-set)
              input (first sample)
              target (second sample)
              output (n/recall net input)
              hit (if (= output target) 1 0)
              error (a/cross-entropy-cost target output)]
          (recur (+ acc hit) (+ errors error) (rest test-set)))
        (vector (/ acc total) (/ errors total))))))

(defn- do-epoche [epoche net training-set test-set]
  (let [trained (n/train net training-set)
        [accuracy error] (test-net trained test-set)]
    (do
      (print "epoche: ")
      (print epoche)
      (print " -> ")
      (print accuracy)
      (print " -> ")
      (println error))
    trained))

(defn- train-net [max-epoche]
  (let [lines (read-lines "resources/mlp-training.dat")
        training-set (make-training-set lines)
        lines2 (read-lines "resources/mlp-test.dat")
        test-set (make-training-set lines2)
        sample (first training-set)
        inodes (count (first sample))
        onodes (count (second sample))
        lrate 0.015
        rf (a/l1-regularization lrate 0.1 (count training-set))
        net (mlp/build [inodes 10 onodes] {:lrate lrate, :momentum 0.25, :rf rf})
        epoche 1]
    (loop [net net
           epoche epoche]
      (if (= max-epoche epoche)
        net
        (recur (do-epoche epoche net training-set test-set) (inc epoche))))))

(defn -main [& args]
  (time
    (train-net 250)))