#!/usr/bin/env bash

#VARIABLES
ans=""

echo -n "Delete done.txt ? (y/n) " ; read ans

if [ $ans == y ]; then
  rm "done.txt"
fi

echo -n "Delete NMSed.txt ? (y/n) " ; read ans

if [ $ans == y ]; then
  rm "NMSed.txt"
fi

echo -n "Delete out files ? (y/n) " ; read ans

if [ $ans == y ]; then
  rm ./out*
fi

echo -n "Clear ./fwd_res ? (y/n) " ; read ans

if [ $ans == y ]; then
  rm ./fwd_res/*
fi

echo -n "Clear ./cat_res ? (y/n) " ; read ans

if [ $ans == y ]; then
  rm ./cat_res/*
fi

echo -n "Clear ./Catalogs ? (y/n) " ; read ans

if [ $ans == y ]; then
  rm ./Catalogs/*
fi

exit 0
