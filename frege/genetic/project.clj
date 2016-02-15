(defproject genetic "0.1.0-SNAPSHOT"
  :description "Genetic Algorithm for frege"
  :url "https://github.com/DirkArnoldt/playground.git"
  :license {:name "Apache License, Version 2.0"
            :url "http://www.apache.org/licenses/LICENSE-2.0"}
  :dependencies [[org.clojure/clojure "1.7.0"]
                 [org.frege-lang/frege "3.23.450-SNAPSHOT"]]
  :repositories {"sonatype" "https://oss.sonatype.org/content/repositories/snapshots/"}
  :plugins [[lein-fregec "3.23.450"]]
  :frege-source-paths ["src" "test"]
  :profiles {:uberjar {:aot :all
                       :prep-tasks ["fregec" "compile"]}})
