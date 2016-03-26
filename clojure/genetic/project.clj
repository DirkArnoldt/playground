(defproject genetic "0.1.0-SNAPSHOT"
  :description "Genetic Algorithm for clojure"
  :url "https://github.com/DirkArnoldt/playground.git"
  :license {:name "Apache License, Version 2.0"
            :url  "http://www.apache.org/licenses/LICENSE-2.0"}
  :dependencies [[org.clojure/clojure "1.7.0"]
                 [org.clojure/tools.cli "0.3.3"]]
  ;  :main genetic.example)
  :main neuron.example
  :profiles {:uberjar {:aot :all}}
  ;:jvm-opts ["-Dcom.sun.management.jmxremote"
  ;           "-Dcom.sun.management.jmxremote.ssl=false"
  ;           "-Dcom.sun.management.jmxremote.authenticate=false"
  ;           "-Dcom.sun.management.jmxremote.port=43210"]
  )
