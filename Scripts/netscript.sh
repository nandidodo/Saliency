#!/bin/bash

#test script to see if this works at all taken from the web, else we'd have to make it and it would be unfortunate af

echo "starting script"
for ip in $(seq 1 254); do ping -c 1 192.168.1.$ip>/dev/null; 
	echo "pinging 192.168.1.$ip";
    [ $? -eq 0 ] && echo "192.168.1.$ip UP" || : ;
done
