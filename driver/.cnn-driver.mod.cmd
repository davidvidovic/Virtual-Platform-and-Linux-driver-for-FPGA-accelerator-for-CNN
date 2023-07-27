cmd_/home/david/Documents/driver/cnn-driver.mod := printf '%s\n'   cnn-driver.o | awk '!x[$$0]++ { print("/home/david/Documents/driver/"$$0) }' > /home/david/Documents/driver/cnn-driver.mod
