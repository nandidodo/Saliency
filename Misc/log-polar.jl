# julia. Okay the aim here is to do the log plotting functionality injulia instead
# for the speed boost


function polar_to_cartesian()
		
	end

function log_polar(image, center::AbstractArray, p_n::Integer = -1, t_n::Integer = -1)
	# I need to figure out what this is. what is i_n, j_n in actually understandable terms?
	#similaly, what are the other things also?
	i_n, j_n = shape(image)[:2] #none of this syntax is going to be right. oh well

	i_c = max(i_0, i_n - i_0)
	j_c = max(j_0, j_n - j_0)
	# this is the distance from the transforms focus to the furthest corner which are i_c and j_c
	# I think they are width and height. Still not sure what i_n, j_n actually are
	d_c = (i_c **2 + j_c **2) ** 0.5

	if p_n == -1
		p_n = ceil(d_c)
	end
	if t_n == -1
		t_n = j_n
	end

	#scale factors along the size of the step along the transforms
	p_s = log(d_c)/p_n # the entirely cryptic variable names are annoying here
	t_s = 2.0 * pi / t_n

	#we create transformed array
	transformed:: Array{AbstractInt} = zeros((p_n, t_n) + shape(image)[2:])

	#scans the transform across coordinate axes, at each step calculates reverse transform
	# if he coordinates fall within boundaries of input image, take cells value into the transform
	for p in 0:p_n
		p_exp = exp(p*p_s)
		for t in 0:t_n
			t_rad = t * t_s
			i = int(i_0 + p_exp * sin(t_rad))
			j = int(j_0 + p_exp * cos(t_rad))

			if 0 <= i < i_n && 0 <=j < j_n
				#so the reverse transformed is inside the image
				transformed[p,t] = image[i,j]
	return transformed
