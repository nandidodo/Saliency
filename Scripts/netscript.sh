#!/bin/bash

#test script to see if this works at all taken from the web, else we'd have to make it and it would be unfortunate af

# the aim is we might eventually be able to use this on the thing so it makes sense really
# i.e. we can check what networks are available for it to see how it works
# but I honestly just really don't know lol... so idk


#initialise our set of ip addresses
ip_addresses=()

#ip_addresses+=("foo")
echo "starting script"
for ip in $(seq 1 254); do
	ip_addr=192.168.1.$ip
	ping -c 1 $ip_addr>/dev/null; 
	echo "pinging $ip_addr";
    [ $? -eq 0 ] && echo "$ip_addr UP" && ip_addresses+=($ip_addr) || echo "$ip_addr down"
	#ip_addresses+=("in loop")
	echo "${ip_addresses[@]}"
done
echo "script finished"
#echo "$ip_addresses"
