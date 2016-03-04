(ns util.matrix)

(defn initialize-matrix [size]
  (vec (map vec (partition size (repeat (* size size) 0.0)))))

(defn build-indexes [si sj]
  (for [i (range si) j (range sj)]
    [i j]))

(defn get-indexes [m]
  (let [[first & rest] m
        si (count m)
        sj (count first)]
    (build-indexes si sj)))

(defn etransform [f m idx]
  (assoc-in m idx (f (get-in m idx))))

(defn etransform-with-index [f m idx]
  (assoc-in m idx (f (get-in m idx) idx)))

(defn transform [f m]
  (let [transformer (partial etransform f)]
    (reduce transformer m (get-indexes m))))

(defn transform-with-index [f m]
  (let [transformer (partial etransform-with-index f)]
    (reduce transformer m (get-indexes m))))
