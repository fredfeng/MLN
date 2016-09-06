#!/bin/bash
BENCHMARKS="
  pjbench/ashesJSuite/benchmarks/javasrc-p
  pjbench/ashesJSuite/benchmarks/schroeder-m
  pjbench/ashesJSuite/benchmarks/toba-s
  pjbench/dacapo/shared
  pjbench/dacapo/benchmarks/antlr
  pjbench/dacapo/benchmarks/luindex
  pjbench/dacapo/benchmarks/lusearch
  pjbench/hedc
  pjbench/weblech-0.0.3"
OUT=~/temp/chord-$(date +%Y%m%d)
mkdir $OUT
cp setenv benchmarks.txt clients.txt $OUT
mkdir $OUT/incubator
cp -r incubator/classes/ incubator/lib/ incubator/runner.pl incubator/scripts/ $OUT/incubator/
mkdir -p $OUT/incubator/src/chord/analyses
cp -r incubator/src/chord/analyses/experiment/ $OUT/incubator/src/chord/analyses/
for F in $(find incubator/src/ -type f | grep -v "java$" | grep -v "rb$"); do
  D=$(dirname $F)
  mkdir -p $OUT/$D
  cp $F $OUT/$F
done
cp -r jchord $OUT
rm -rf $OUT/jchord/.git
for B in $BENCHMARKS; do
  U=$(dirname $B)
  mkdir -p $OUT/$U
  cp -r $B $OUT/$U
done
# todo: rm this script
