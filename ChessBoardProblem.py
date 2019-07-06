def Steps(whiteX, whiteY, blackX, blackY): 
	c=max(abs(whiteX - blackX), abs(whiteY - blackY))
	print("Minimum steps required",c) 
	while ((whiteX != blackX) or (whiteY != blackY)): 
		print("\nwhite\n")			# White pawn

		if (whiteX < blackX): 			# Go Up 
			print('U',end = "") 
			whiteX += 1
		
		if (whiteX > blackX): 			# Go Down
			print('D',end = "") 
			whiteX -= 1
		
		if (whiteY > blackY): 			# Go right 
			print('R') 
			whiteY -= 1
		 
		if (whiteY < blackY): 			# Go left

			print('L',end = "") 
			whiteY += 1
 
		print("\nblack\n")			# Black pawn

		if (whiteX < blackX):			# Go up 
			print('D',end = "") 
			blackX -= 1
		
		if (whiteX > blackX): 			# Go down 
			print('U',end = "") 
			blackX += 1
		
		if (whiteY > blackY): 			# Go left 
			print('L') 
			blackY += 1
		 
		if (whiteY < blackY): 			# Go right
			print('R',end = "") 
			blackY -= 1
whiteX = 1
whiteY = 1
blackX = 2
blackY = 5
Steps(whiteX, whiteY, blackX, blackY) 
