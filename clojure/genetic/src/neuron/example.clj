(ns neuron.example
  (:gen-class)
  (:require [neuron.core :as n]
            [neuron.hopfield :as h]
            [neuron.mlp :as mlp])
  (:require [util.activation :as a]
            [util.cost :as cost]
            [util.regularization :as reg])
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

(defn max-item [a]
  (first (reduce #(if (> (second %1) (second %2)) %1 %2) (map-indexed #(vector %1 %2) a))))

(defn hit [a b]
  (let [amax (max-item a)
        bmax (max-item b)]
    (= amax bmax)))

;------------- cost functions

(defn quadratic-cost [target output]
  (* 0.5 (reduce + 0.0 (mapv cost/squared-error output target))))

(defn cross-entropy-cost [target output]
  (* -1 (reduce + (mapv cost/log-likelihood output target))))


(defn- test-net [net test-set]
  (let [total (count test-set)]
    (loop [acc 0.0
           errors 0.0
           test-set test-set]
      (if (seq test-set)
        (let [sample (first test-set)
              input (first sample)
              target (second sample)
              output (n/recall net input)
              hit (if (hit output target) 1 0)
              error (cross-entropy-cost target output)]
          (recur (+ acc hit) (+ errors error) (rest test-set)))
        (vector (/ acc total) (/ errors total))))))

(def max-y 500.0)

(defn scale [y]
  (- max-y (Math/min max-y (* 1000.0 y))))

(defn scalep [y]
  (- max-y  (* max-y y)))

(defn- do-epoche [gfx epoche net training-set test-set]
  (let [trained (n/train net (shuffle training-set))
        [train-accuracy train-error] (test-net trained training-set)
        [accuracy error] (test-net trained test-set)]
    (do
      (.setColor gfx (java.awt.Color. 255 0 0))
      (.fillRect gfx epoche (scalep train-accuracy) 2 2)
      (.setColor gfx (java.awt.Color. 0 255 0))
      (.fillRect gfx epoche (scalep accuracy) 2 2)
      (.setColor gfx (java.awt.Color. 127 127 127))
      (.fillRect gfx epoche (scale train-error) 2 2)
      (.setColor gfx (java.awt.Color. 0 0 255))
      (.fillRect gfx epoche (scale error) 2 2)
      (print "epoche: ")
      (print epoche)
      (print " -> ")
      (print train-accuracy)
      (print " / ")
      (print accuracy)
      (print " : ")
      (print train-error)
      (print " / ")
      (println error))
    trained))

(defn- make-gfx [x y]
  (let [frame (doto (java.awt.Frame.)
                            (.setSize (java.awt.Dimension. x (+ 50 y)))
                            (.setVisible true))
        gfx (.getGraphics frame)]
    (.setColor gfx (java.awt.Color. 255 255 255))
    (.fillRect gfx x y x y)
    [frame gfx]))

(defn- train-net [max-epoche]
  (let [lines (read-lines "resources/mlp-training.dat")
        training-set (shuffle (make-training-set lines))
;        lines2 (read-lines "resources/mlp-test.dat")
        ;test-set (take 15 training-set)
        sample (first training-set)
        inodes (count (first sample))
        onodes (count (second sample))
        batch-size 10
        lrate 0.003
        momentum 0.3
        rparam (/ 0.01 (count training-set))
        rf (reg/l2-regularization lrate rparam)
        net (mlp/build [inodes 10 onodes] {:lrate lrate, :momentum momentum,
                                           :rf rf, :af a/relu, :af' a/relu',
                                           :batch-size batch-size})
        epoche 1
        [frame gfx] (make-gfx max-epoche max-y)]
    (loop [net net
           epoche epoche]
      (if (= max-epoche epoche)
        (.dispose frame)
        (recur (do-epoche gfx epoche net (drop 15 training-set) (take 15 training-set)) (inc epoche))))))

(defn -main [& args]
  (time
    (train-net 1000)))