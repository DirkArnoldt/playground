(ns neuron.mlp
  (:require [neuron.core])
  (:require [util.matrix :as m])
  (:require [util.activation :as a])
  (:import (neuron.core NeuralNet)))

(def LEARNING-RATE 0.025)
(def MOMENTUM 0.25)
(def REGULARIZATION-PARAM 5.0)
(def BATCH-SIZE 10)

(defprotocol Layer
  (train-layer [this input target])
  (feed [this input])
  (finish-batch [this m]))

;---

(defn- calc-net-inputs [weights bias input]
  (mapv #(+ (m/dot %1 input) %2) (m/transpose weights) bias))

(defn- calc-outputs [af net-input]
  (vec (map #(af %) net-input)))

(defn- calc-activations [af weights bias input]
  (let [net-inputs (calc-net-inputs weights bias input)
        outputs (calc-outputs af net-inputs)]
    (vector net-inputs outputs)))

(defn- calc-back-errors [weights deltas]
  (vec (map #(m/dot % deltas) weights)))

(defn- calc-output-deltas [af' net-inputs errors]
  (mapv #(* (af' %1) %2) net-inputs errors))

(defn- calc-output-errors [output target]
  (m/sub output target))

(defn- costf-quadratic [af' net-inputs output target]
  (let [errors (calc-output-errors output target)]
    (calc-output-deltas af' net-inputs errors)))

(defn- costf-cross-entropy [af' net-inputs output target]
  (let [errors (calc-output-errors output target)]
    errors))

(defn- calc-deltas [next cf af' net-inputs output target]
  (if (nil? next)
    (vector nil (cf af' net-inputs output target))
    (let [[layer errors] (train-layer next output target)
          deltas (calc-output-deltas af' net-inputs errors)]
      (vector layer deltas))))

(defn- calc-weight-deltas [eta input deltas momentum dweights]
  (let [si (count input)
        sj (count deltas)
        weight-deltas (for [i (range si) j (range sj)]
                        (+ (* eta (deltas j) (input i))
                           (* momentum (get-in dweights [i j]))))]
    (m/to-matrix sj weight-deltas)))

(defn- calc-bias-deltas [lr deltas momentum dbias]
  (mapv #(+ (* lr %1)
            (* momentum %2)) deltas dbias))

(defn- update-weights [weights deltas m rf]
  (m/transform-with-index (fn [w idx] (- (rf w) (* m (get-in deltas idx))) ) weights))

(defn- update-bias [bias deltas m]
  (mapv #(- %1 (* m %2)) bias deltas))

;---

(defrecord BLayer [weights dw_batch dw_t-1 bias db_batch db_t-1 options next]
  Layer
  (train-layer [_ input target]
    (let [af (:af options)
          af' (:af' options)
          cf (:cf options)
          eta (:lrate options)
          momentum (:momentum options)
          [net-input output] (calc-activations af weights bias input)
          [layer deltas] (calc-deltas next cf af' net-input output target)
          back-errors (calc-back-errors weights deltas)
          dw_t (calc-weight-deltas eta input deltas momentum dw_t-1)
          dw_batch (m/mapm + dw_batch dw_t)
          db_t (calc-bias-deltas eta deltas momentum db_t-1)
          db_batch (m/add db_batch db_t)]
      [(BLayer. weights dw_batch dw_t bias db_batch db_t options layer) back-errors]))
  (feed [_ input]
    (let [af (:af options)
          [_ output] (calc-activations af weights bias input)]
      (if (nil? next)
        output
        (feed next output))))
  (finish-batch [_ m]
    (let [rf (:rf options)
          batch-factor (/ 1 m)
          weights (update-weights weights dw_batch batch-factor rf)
          bias (update-bias bias db_batch batch-factor)
          dw_batch (m/zero-matrix (m/dimension dw_batch))
          db_batch (m/zero-vector (count db_batch))
          layer (if (nil? next) nil (finish-batch next m))]
      (BLayer. weights dw_batch dw_t-1 bias db_batch db_t-1 options layer))))

;
;(defn- feedforward [layer input target]
;  (let [options (:options layer)
;        af (:af options)
;        af' (:af' options)
;        cf (:cf options)
;        weights (:weights layer)
;        bias (:bias layer)
;        [net-inputs output] (calc-activations af weights bias input)]
;    (calc-deltas (:next layer) cf af' net-inputs output target)))
;
;(defrecord GLayer [weights dweights bias dbias options next]
;  Layer
;  (train-layer [this input target]
;    (let [lrate (:lrate options)
;          momentum (:momentum options)
;          rf (:rf options)
;          ; feedforward
;          [layer deltas] (feedforward this input target)
;          ; backpropagate
;          back-errors (calc-back-errors weights deltas)
;          weight-deltas (calc-weight-deltas lrate input deltas momentum dweights)
;          bias-deltas (calc-bias-deltas lrate deltas momentum dbias)
;          new-weights (update-weights weights weight-deltas 1 rf)
;          new-bias (update-bias bias bias-deltas 1)]
;      [(GLayer. new-weights weight-deltas new-bias bias-deltas options layer) back-errors]))
;  (feed [_ input]
;    (let [af (:af options)
;          [_ outputs] (calc-activations af weights bias input)]
;      (if (nil? next)
;        outputs
;        (feed next outputs)))))
;

(defn- train-batch [layer batch batch-size]
  (loop [layer layer
         batch batch
         batch-size batch-size]
    (if (seq batch)
      (let [[input target] (first batch)
            trained (first (train-layer layer input target))]
        (recur trained (rest batch) batch-size))
      (finish-batch layer batch-size))))

(defrecord MultiLayerPerceptron [layer batch-size]
  NeuralNet
  (train [_ trainings-set]
    (loop [batches (partition batch-size trainings-set)
           current layer]
      (if (seq batches)
        (recur (rest batches) (train-batch current (first batches) batch-size))
        (MultiLayerPerceptron. current batch-size))))
  (recall [_ pattern]
    (feed layer pattern)))

(defn- build-layer [coll options]
  (let [[i j & remain] coll
        dim (vector i j)
        bias (m/compute-vector j a/rand-gaussian)
        dbias (m/zero-vector j)
        weights (m/compute-matrix dim (a/randit i))
        dweights (m/zero-matrix dim)]
    (if (seq remain)
      (let [next (build-layer (rest coll) options)]
        (BLayer. weights dweights dweights bias dbias dbias options next))
      (BLayer. weights dweights dweights bias dbias dbias options nil))))

(defn build [coll options]
  (let [{:keys [lrate momentum batch-size af af' rf cf],
         :or {lrate LEARNING-RATE, momentum MOMENTUM, batch-size BATCH-SIZE
              af a/sigmoid, af' a/sigmoid', rf identity, cf costf-cross-entropy}} options
        layer-options {:lrate lrate, :momentum momentum, :af af, :af' af', :rf rf, :cf cf}]
    (MultiLayerPerceptron. (build-layer coll layer-options) batch-size)))
