ls -l pattern/|awk '{print $9}'|grep png|sed 's/\.png//g'
