import numpy as np

# np.zeros((n, n)) - функция, которая возвращает матрицу размера n на n, заполненную нулями
# np.zeros(n) - функция, которая возвращает массив, заполненный нулями

# функция, которая возвращает значение полинома (построенного из коэффициентов массива a) при заданном значении x
def polynom(a, x):
	result = 0
	for i in range(a.size):
		result += a[i] * (x ** (a.size - i - 1))
	return result

# функция, которая возвращает знак числа x
def sign(x):
	if x > 0:
		return 1
	elif x < 0:
		return -1
	else:
		return 0

# функция, которая возвращает корни (с погрешностью error) полинома, составленного по коэффициентам массива a
def Lobachevski_Greffe_method(a, error):
	a_temp = np.zeros(a.size, dtype = np.float128)
	for i in range(a.size):
		a_temp[i] = a[i]
	a_temp_0 = np.zeros(a.size, dtype = np.float128)
	a_next = np.zeros(a.size, dtype = np.float128)
	for i in range(a.size):
		a_next[i] = a[i]
	a_status = np.zeros(a.size, dtype = int)
	need_while = True
	count = 0
	#print(count, ': ', a_next)
	while(need_while):
		for i in range(a.size):
			k = 1
			add = 0
			while((i - k >= 0) and (i + k < a.size) and (k <= i)):
				add += ((-1) ** k) * a_temp[i - k] * a_temp[i + k]
				k += 1
			add *= 2
			a_next[i] *= a_next[i]
			a_next[i] += add
			if(sign(a_next[i]) == 0):
				a_temp_0[i] = a_temp[i]
			elif(sign(a_temp[i] != 0)):
				a_temp_0[i] = 0
			if(((sign(a_next[i]) != sign(a_temp[i])) and (sign(a_next[i]) != sign(a_temp_0[i]))) and (sign(a_next[i]) != 0)):
				if(a_status[i] <= 0):
					a_status[i] -= 1
					if(a_status[i] < -3):
						a_status[i] = 3
			elif(abs(a_next[i] - a_temp[i] ** 2) <= error):
				a_status[i] = 1
			elif(abs(a_next[i] - (a_temp[i] ** 2) / 2) <= error):
				a_status[i] = 2
		for i in range(a.size):
			a_temp[i] = a_next[i]
		need_while = False
		for i in range(a.size):
			if (a_status[i] <= 0):
				need_while = True
				break
		count += 1
		#print(count, ': ', a_next)
	i = 1
	roots = np.zeros(a.size - 1, dtype = complex)
	complex_first = np.array([], dtype = int)
	while(i < a.size):
		if(a_status[i] == 1):
			roots[i - 1] = abs(a_next[i] / a_next[i - 1]) ** (1 / (2 ** count))
			if (abs(polynom(a, roots[i - 1])) > error):
				roots[i - 1] *= -1
		elif(a_status[i] == 2):
			roots[i - 1] = abs(a_next[i] / a_next[i - 1]) ** (1 / (2 ** (count + 1)))
			if (abs(polynom(a, roots[i - 1])) > error):
				roots[i - 1] *= -1
			roots[i] = roots[i - 1]
			i += 1
		elif(a_status[i] == 3):
			complex_first = np.append(complex_first, i - 1)
			roots[i - 1] = abs(a_next[i + 1] / a_next[i - 1]) ** (1 / (2 ** (count + 1)))
			i += 1
		i += 1
	if(complex_first.size > 0):
		j = 1
		const = -(a[1] / a[0])
		while(j < a.size):
			if(a_status[j] == 1):
				const -= roots[j - 1]
			elif(a_status[j] == 2):
				const -= 2 * roots[j - 1]
				j += 1
			elif(a_status[j] == 3):
				j += 1
			j += 1
		for k in range(complex_first.size):
			temp = const / (2 * roots[complex_first[k]])
			roots[complex_first[k]] = complex(roots[complex_first[k]].real * temp, roots[complex_first[k]].real * ((1 - temp ** 2) ** (1 / 2)))
			roots[complex_first[k] + 1] = complex(roots[complex_first[k]].real, -roots[complex_first[k]].imag)
	#print()
	return roots

# функция, которая возвращает решение системы линейных уравнений Ax=B
def Linear_equations_system_solve(A, B):
	n = A[0].size
	a = np.zeros((n, n))
	b = np.zeros(n)
	for i in range(n):
		b[i] = B[i]
		for j in range(n):
			a[i][j] = A[i][j]
	for i in range(n - 1):
		for j in range(i + 1, n):
			if (a[j][i] != 0):
				for k in range(i + 1, n):
					a[j][k] -= a[i][k] * a[j][i] / a[i][i]
				b[j] -= b[i] * a[j][i] / a[i][i]
		for k in range(i + 1, n):
			a[k][i] = 0
	temp = n - 1
	result = np.zeros(n)
	result[temp] = b[temp] / a[temp][temp]
	for i in range(1, n + 1):
		index = n - i
		temp = b[index]
		for j in range(index + 1, n):
			temp -= a[index][j] * result[j]
		result[index] = temp / a[index][index]
	return result

# Проверка линейных векторов из массива векторов C на линейную независимость
def Vectors_independence_check(C):
	k = C[0].size
	n = (int)(C.size / k)
	#print(C)
	if (k > n):
		return False
	a = np.zeros((k, n))
	for i in range(k):
		for j in range(n):
			a[i][j] = C[j][i]
	#print(a)
	for i in range(k - 1):
		for j in range(i + 1, k):
			if (a[j][i] != 0):
				for l in range(i + 1, n):
					a[j][l] -= a[i][l] * a[j][i] / a[i][i]
		for l in range(i + 1, k):
			a[l][i] = 0
	#print(a)
	for i in range(k - 1, n):
		if (a[k - 1][i] != 0):
			return True
	return False

# Регулярный метод Данилевского
def Danilevski_regular_method(A, error):
	a = np.zeros((A[0].size, A[0].size)) # матрица A, подвераемая преобразованию (матрица Ф)
	m = np.zeros((A[0].size, A[0].size)) # итоговая матрица преобразования S
	n = a[0].size
	#print(A)
	for i in range(n):
		m[i][i] = 1 # формирование единичный матрицы
		for j in range(n):
			a[i][j] = A[i][j] # копирование матрицы A
	for i in range(n - 1):
		a_line_temp = np.zeros(a[0].size)
		# если a[n - i - 1][n - i - 2] равно нулю, то меняем местами столбцы (n - i - 2 на j, где a[j][n - i - 2] не равно нулю)
		if (a[n - i - 1][n - i - 2] == 0):
			j = 0
			while (j < n - i - 2 and a[n - i - 1][j] == 0):
				j += 1
			if (j == n - i - 2):
				continue #
			for k in range(n):
				temp = a[i][n - i - 2]
				a[i][n - i - 2] = a[j][n - i - 2]
				a[j][n - i - 2] = temp
		for j in range(n):
			a_line_temp[j] = a[n - i - 1][j]
		
		coefficient = a[n - i - 1][n - i - 2]
		m_temp = np.zeros((n, n)) # матрица преобразования M
		m_prev = np.zeros((n, n))
		# формирование матрицы преобразования M
		for j in range(n):
			m_temp[j][j] = 1
			m_temp[n - i - 2][j] = -a[n - i - 1][j] / coefficient
			for k in range(n):
				m_prev[j][k] = m[j][k] # копирвоание матрицы m
		m_temp[n - i - 2][n - i - 2] = 1 / coefficient
		# умножение матриц m на m_temp (формирование итоговой матрицы преобразования S)
		m = np.zeros((n, n))
		for j in range(n):
			for k in range(n):
				for l in range(n):
					m[j][k] += m_prev[j][l] * m_temp[l][k]
		for j in range(n):
			a[j][n - i - 2] /= coefficient
			
		a_temp = np.zeros((A[0].size, A[0].size))
		for q in range(n):
			for w in range(n):
				a_temp[q][w] = a[q][w] # копирование матрицы a
		
		# преобразование матрицы A вида: A'=A*M
		for j in range(n):
			for k in range(n - i - 2):
				a[j][k] -= a_temp[j][n - i - 2] * a_temp[n - i - 1][k]
			for k in range(n - i - 1, n):
				a[j][k] -= a_temp[j][n - i - 2] * a_temp[n - i - 1][k]
		
		for q in range(n):
			for w in range(n):
				a_temp[q][w] = a[q][w] # копирование матрицы a
		
		# преобразование матрицы A' вида: A''=M^(-1)*A'
		for j in range(n):
			a[n - i - 2][j] = 0
			for k in range(n):
				a[n - i - 2][j] += a_line_temp[k] * a_temp[k][j]
		#print(a)
	
	our_polynom = np.zeros(n + 1) # массив коэффициентов полинома
	our_polynom[0] = 1
	for i in range(n):
		our_polynom[i + 1] = -a[0][i] # занесение коэффициентов полинома
	#print(our_polynom)
	
	temp = Lobachevski_Greffe_method(our_polynom, error) # нахождение корней полинома
	eigenvalues = np.zeros(n) # массив собственных значений
	
	# занесение собственных значений в массив и их вывод
	print('Danilevski regular method:\n\tEigenvalues:')
	for i in range(n):
		eigenvalues[i] = temp[i].real
		print('\t\t', i + 1, ': ', eigenvalues[i])
	#print(eigenvalues)
	
	eigenvectors = np.zeros((n, n)) # массив собственных векторов (матрица)
	for i in range(n):
		# формирование собственного вектора матрицы Ф, соответствующий i-ому собственному значению
		temp2 = np.zeros(n)
		for j in range(n):
			temp2[j] = eigenvalues[i] ** (n - j - 1)
		# формирование собственного вектора матрицы A, соответствующий i-ому собственному значению
		for j in range(n):
			for k in range(n):
				eigenvectors[j][i] += m[j][k] * temp2[k]
	# вывод собственных векторов
	print('\tEigenvectors:\n\t', end = "")
	for i in range(n):
		print('\t', '   {0}   '.format(i + 1), end = "")
	print()
	for i in range(n):
		print('\t\t', end = "")
		for j in range(n):
			print('{0:f}'.format(eigenvectors[i][j]), '\t', end = "")
		print()
	print()
	#print(eigenvectors)
	
	return np.insert(eigenvectors, 0, eigenvalues, 0)

# Нерегулярный метод Данилевского
def Danilevski_nonregular_method(A, error):
	a = np.zeros((A[0].size, A[0].size)) # матрица A, подвергаемая преобразованию (матрица Ф)
	m = np.zeros((A[0].size, A[0].size)) # матрица преобразований S
	n = a[0].size
	#print(A)
	for i in range(n):
		m[i][i] = 1 # формирование единичной матрицы
		for j in range(n):
			a[i][j] = A[i][j] # копирование матрицы A
	
	for i in range(n - 1):
		m_temp = np.zeros((n, n)) # матрица преобразований M
		# формирование матрицы преобразований M
		for j in range(1, n):
			m_temp[j][j - 1] = 1
			m_temp[j - 1][n - 1] = a[j - 1][n - 1]
		m_temp[n - 1][n - 1] = a[n - 1][n - 1]
		#print(m_temp)
		
		m_prev = np.zeros((n, n))
		for j in range(n):
			for k in range(n):
				m_prev[j][k] = m[j][k] # копирование матрицы m
		
		# умножение матриц m на m_temp (формирование итоговой матрицы преобразования S)
		m = np.zeros((n, n))
		for j in range(n):
			for k in range(n):
				for l in range(n):
					m[j][k] += m_prev[j][l] * m_temp[l][k]
		#print(m)
			
		a_temp = np.zeros((n, n))
		for q in range(n):
			for w in range(n):
				a_temp[q][w] = a[q][w] # копирование матрицы a
		# умножение матриц a на m_temp (A'=A*M)
		a = np.zeros((n, n))
		for j in range(n):
			for k in range(n):
				for l in range(n):
					a[j][k] += a_temp[j][l] * m_temp[l][k]
		#print(a)
		
		# формирование матрицы M^(-1)
		m_temp = np.zeros((n, n))
		for j in range(1, n):
			m_temp[j - 1][j] = 1
			m_temp[j - 1][0] = -a_temp[j][n - 1] / a_temp[0][n - 1]
		m_temp[n - 1][0] = 1 / a_temp[0][n - 1]
		#print(m_temp)
		
		for q in range(n):
			for w in range(n):
				a_temp[q][w] = a[q][w] # копирование матрицы a
		# умножение матриц m_temp на a (A''=M^(-1)*A)
		a = np.zeros((n, n))
		for j in range(n):
			for k in range(n):
				for l in range(n):
					a[j][k] += m_temp[j][l] * a_temp[l][k]
		#print(a)
	
	our_polynom = np.zeros(n + 1) # массив коэффициентов полинома
	our_polynom[0] = 1
	for i in range(n):
		our_polynom[i + 1] = -a[n - i - 1][n - 1] # занесение коэффициентов полинома
	#print(our_polynom)
	
	temp = Lobachevski_Greffe_method(our_polynom, error) # нахождение корней полинома
	eigenvalues = np.zeros(n) # массив собственных значений
	
	# занесение собственных значений в массив и их вывод
	print('Danilevski nonregular method:\n\tEigenvalues:')
	for i in range(n):
		eigenvalues[i] = temp[i].real
		print('\t\t', i + 1, ': ', eigenvalues[i])
	#print(eigenvalues)
	
	eigenvectors = np.zeros((n, n)) # массив собственных векторов (матрица)
	for i in range(n):
		# формирование собственного вектора матрицы Ф, соответствующий i-ому собственному значению
		temp2 = np.zeros(n)
		temp2[n - 1] = 1
		for j in range(1, n):
			temp2[n - j - 1] = eigenvalues[i] ** j
			for k in range(j):
			    temp2[n - j - 1] += our_polynom[k + 1] * eigenvalues[i] ** (j - k - 1)
		# формирование собственного вектора матрицы А, соответствующий i-ому собственному значению
		for j in range(n):
			for k in range(n):
				eigenvectors[j][i] += m[j][k] * temp2[k]
		# выравнивание собственного вектора по последней координате (последняя координата будет равняться 1)
		for j in range(n):
		    eigenvectors[j][i] /= eigenvectors[n - 1][i]
	# вывод собственных векторов
	print('\tEigenvectors:\n\t', end = "")
	for i in range(n):
		print('\t', '   {0}   '.format(i + 1), end = "")
	print()
	for i in range(n):
		print('\t\t', end = "")
		for j in range(n):
			print('{0:f}'.format(eigenvectors[i][j]), '\t', end = "")
		print()
	print()
	#print(eigenvectors)
	
	return np.insert(eigenvectors, 0, eigenvalues, 0)

# Метод Леверье
def Leverie_method(A, error):
	a = np.zeros((A[0].size, A[0].size, A[0].size)) # массив матриц A^k (k от 1 до n)
	n = a[0][0].size
	#print(A)
	for i in range(n):
		for j in range(n):
			a[0][i][j] = A[i][j] # копирование матрицы A
	
	# умножение матриц A^i на A
	for i in range(n - 1):
		for j in range(n):
			for k in range(n):
				for l in range(n):
					a[i + 1][j][k] += a[i][j][l] * a[0][l][k]
		#print(a[i + 1])
	
	# формирование коэффициентов Sk
	s = np.zeros(n)
	for i in range(n):
		for j in range(n):
			s[i] += a[i][j][j]
	
	our_polynom = np.zeros(n + 1) # массив коэффициентов полинома
	# формирование коэффициентов полинома
	our_polynom[0] = 1
	for i in range(n):
		our_polynom[i + 1] = s[i]
		for j in range(i):
			our_polynom[i + 1] -= s[i - j - 1] * our_polynom[j + 1]
		our_polynom[i + 1] /= i + 1
	for i in range(n):
		our_polynom[i + 1] *= -1
	#print(our_polynom)
	#eigenvalues = np.roots(our_polynom)
	
	temp = Lobachevski_Greffe_method(our_polynom, error) # нахождение корней полинома
	eigenvalues = np.zeros(n) # массив собственных значеий
	# занесение собственных значений и их вывод
	print('Leverie method:\n\tEigenvalues:')
	for i in range(n):
		eigenvalues[i] = temp[i].real
		print('\t\t', i + 1, ': ', eigenvalues[i])
	#print(eigenvalues)
	
	eigenvectors = np.zeros((n, n)) # массив собственных векторов (матрица)
	a_temp = np.zeros((n, n)) # матрица вида: A-л(k)E
	for k in range(n):
		# формирование матрицы вида A-л(k)E
		for i in range(n):
			for j in range(n):
				a_temp[i][j] = A[j][i]
			a_temp[i][i] -= eigenvalues[k]
		#print(a_temp)
		# составление системы линейных уравнений Mx=b размерности n-1
		M = np.zeros((n - 1, n - 1))
		b = np.zeros(n - 1)
		for i in range(n - 1):
			b[i] = -a_temp[n - 1][i + 1]
			for j in range(n - 1):
				M[i][j] = a_temp[i + 1][j]
		
		eig_temp = Linear_equations_system_solve(M, b) # вычисление решения системы линейных уравнений Mx=b
		
		# формирование собственного вектора A, соответствующий i-ому собственному значению, с последней координатой равной единице
		eigenvectors[n - 1][k] = 1
		for i in range(n - 1):
			eigenvectors[i][k] = eig_temp[i]
	
	# вывод собственных векторов
	print('\tEigenvectors:\n\t', end = "")
	for i in range(n):
		print('\t', '   {0}   '.format(i + 1), end = "")
	print()
	for i in range(n):
		print('\t\t', end = "")
		for j in range(n):
			print('{0:f}'.format(eigenvectors[i][j]), '\t', end = "")
		print()
	#print(eigenvectors)
	print()
	
	return np.insert(eigenvectors, 0, eigenvalues, 0)

# Метод Фадеева
def Fadeev_method(A, error):
	a = np.zeros((A[0].size + 1, A[0].size, A[0].size)) # массив матриц Ak
	b = np.zeros((A[0].size + 1, A[0].size, A[0].size)) # массив матриц Bk
	n = a[0][0].size
	q = np.zeros(n) # массив будущих коэффициентов полинома
	
	#print(A)
	for i in range(n):
		b[0][i][i] = 1 # формирование единичной матрицы
		for j in range(n):
			a[0][i][j] = A[i][j] # копирование матрицы A
	
	for i in range(n):
		# умножение матриц A и Bk (Ak=A*Bk)
		for j in range(n):
			for k in range(n):
				for l in range(n):
					a[i][j][k] += a[0][j][l] * b[i][l][k]
				if(i == 0):
					a[0][j][k] = A[j][k]
		
		# вычисление коэффициентов полинома
		for j in range(n):
			q[i] += a[i][j][j] # вычисление следа матрицы Ai
		q[i] /= i + 1
		
		# формирование матрицы B(i+1) (B(i+1)=Ai-лE)
		for j in range(n):
			for k in range(n):
				b[i + 1][j][k] = a[i][j][k]
		for j in range(n):
			b[i + 1][j][j] -= q[i]
		#print(a[i])
	
	our_polynom = np.zeros(n + 1) # массив коэффициентов полинома
	our_polynom[0] = 1
	for i in range(n):
		our_polynom[i + 1] = -q[i] # занесение коэффициентов полинома
	#print(our_polynom)
	
	temp = Lobachevski_Greffe_method(our_polynom, error) # вычисление корней полинома
	eigenvalues = np.zeros(n) # массив собственных значений
	# занесение собственных значений и их вывод
	print('Fadeev method:\n\tEigenvalues:')
	for i in range(n):
		eigenvalues[i] = temp[i].real
		print('\t\t', i + 1, ': ', eigenvalues[i])
	#print(eigenvalues)
	
	eigenvectors = np.zeros((n, n)) # массив собственнных векторов (матрица)
	for i in range(n):
		Q = np.zeros((n, n)) # вспомгательная матрица Q=(лi^(n-1))E + (лi^(n-2))B1 + (лi^(n-3))B2 + ... + B(n-1)
		# формирование матрицы Q
		for l in range(n):
			for j in range(n):
				for k in range(n):
					Q[j][k] += eigenvalues[i] ** (n - l - 1) * b[l][j][k]
		#print(Q)
		
		# формирование собственного вектора
		for j in range(n):
			eigenvectors[j][i] = Q[j][0] / Q[n - 1][0]
	
	# вывод собственных векторов
	print('\tEigenvectors:\n\t', end = "")
	for i in range(n):
		print('\t', '   {0}   '.format(i + 1), end = "")
	print()
	for i in range(n):
		print('\t\t', end = "")
		for j in range(n):
			print('{0:f}'.format(eigenvectors[i][j]), '\t', end = "")
		print()
	print()
	#print(eigenvectors)
	
	return np.insert(eigenvectors, 0, eigenvalues, 0)

# Метод Крылова
def Krilov_method(A, error):
	a = np.zeros((A[0].size + 1, A[0].size, A[0].size)) # массив матриц A^k
	n = a[0][0].size
	#print(A)
	for i in range(n):
		for j in range(n):
			a[0][i][j] = A[i][j] # копирование матрицы A
	
	# формирование матриц A^i
	for i in range(1, n + 1):
		for j in range(n):
			for k in range(n):
				for l in range(n):
					a[i][j][k] += a[0][j][l] * a[i - 1][l][k]
	
	# нахождение коэффициентов полинома
	count = 0
	j = 0
	while (count < n):
		C = np.zeros((n + 1, n)) # массив векторов Ck
		C[0][j] = 1
		independent = True
		k = 0
		while (independent and k < n):
			# формирование вектора Ck
			for l in range(n):
				for m in range(n):
					C[k + 1][l] += a[k][l][m] * C[0][m]
			
			# проверка системы векторов Ck на линейную зависимость
			if (k >= 1 and k < n - 1):
				C_temp = np.zeros((n, k + 2))
				for l in range(k + 2):
					for m in range(n):
						C_temp[m][l] = C[l][m]
				independent = Vectors_independence_check(C_temp) # проверка векторов на линейную независимость
			k += 1
		count += k
		j += 1
		#print(C)
	
	# составление системы линейных уравнений, решением которого являются коэффициента полинома
	C_temp = np.zeros((n, k))
	for l in range(k):
		for m in range(n):
			C_temp[m][l] = C[l][m]
	
	q = Linear_equations_system_solve(C_temp, C[k]) # нахождение коэффициентов полинома
	#print(q)
	
	our_polynom = np.zeros(n + 1) # массив коэффициентов полинома
	our_polynom[0] = 1
	for i in range(n):
		our_polynom[i + 1] = -q[n - i - 1] # занесение коэффициентов полинома
	#print(our_polynom)
	
	temp = Lobachevski_Greffe_method(our_polynom, error) # нахождение собственных значений
	eigenvalues = np.zeros(n) # массив собственных значений
	# занесение собственных значений и их вывод
	print('Krilov method:\n\tEigenvalues:')
	for i in range(n):
		eigenvalues[i] = temp[i].real
		print('\t\t', i + 1, ': ', eigenvalues[i])
	#print(eigenvalues)
	
	eigenvectors = np.zeros((n, n)) # массив собственных векторов (матрица)
	B = np.zeros((n, n)) # массив коэффициентов линейной комбинации векторов Ck (bi1*C1+bi2*C2+...+bin*Cn), которая равна собственному вектору, соответствующий i-ому собственному значению (столбцы - коэффициенты линейной комбинации, строки соответствуют i-ому собственному значению)
	# формирование матрицы B
	for j in range(n):
		B[j][0] = 1
		for k in range(1, n):
			B[j][k] = eigenvalues[j] ** k
			for l in range(1, k + 1):
				B[j][k] -= eigenvalues[j] ** (k - l) * q[n - l]
	
	# нахождение собственных векторов матрицы A
	for i in range(n):
		# нахождение линейной комбинации bi1*C1+...+bin*Cn
		for j in range(n):
			for k in range(n):
				eigenvectors[j][i] += B[i][k] * C[n - k - 1][j]
		
		# выравнивание собственного вектора по последней координате (последняя координата будет равняться 1)
		for j in range(n):
			eigenvectors[j][i] /= eigenvectors[n - 1][i]
		
	# вывод собственных векторов
	print('\tEigenvectors:\n\t', end = "")
	for i in range(n):
		print('\t', '   {0}   '.format(i + 1), end = "")
	print()
	for i in range(n):
		print('\t\t', end = "")
		for j in range(n):
			print('{0:f}'.format(eigenvectors[i][j]), '\t', end = "")
		print()
	print()
	#print(eigenvectors)
	
	return np.insert(eigenvectors, 0, eigenvalues, 0)

# Метод Вращений
def Rotation_method(A, error):
	a = np.zeros((A[0].size, A[0].size)) # матрица A, подвергаемая преобразованию
	n = a[0].size
	v = np.zeros((n, n)) # итоговая матрица преобразования
	for i in range(n):
		v[i][i] = 1 # формирование единичной матрицы
		for j in range(n):
			a[i][j] = A[i][j] # копирование матрицы A
	
	temp = error + 1 # сумма квадратов недиагональных элементов матрицы A
	#print(A)
	
	# приближённое приведение матрицы к диагональному виду
	while(temp > error / 100000):
		max_a = np.zeros(2, dtype = int) # координаты максимального по модулю недиагонального элемента матрицы A
		max_a[1] = 1
		v_temp = np.zeros((n, n)) # матрица вращения
		temp = 0
		for i in range(n):
			v_temp[i][i] = 1 # формирование единичной матрицы
			
			# поиск максимального по модулю недиагонального элемента матрицы A
			for j in range(n):
				if (i < j and abs(a[i][j]) > abs(a[max_a[0]][max_a[1]])):
					max_a[0] = i
					max_a[1] = j
				if (i != j):
					temp += a[i][j] ** 2 # подсчёт суммы квадратов недиагональных элементов матрицы A
		
		# вычисление cos(fi) и sin(fi) (коэффициентов вращения)
		cos_fi = ((1 + (1 + (2 * a[max_a[0]][max_a[1]] / (a[max_a[0]][max_a[0]] - a[max_a[1]][max_a[1]])) ** 2) ** -0.5) / 2) ** 0.5
		sin_fi = sign((2 * a[max_a[0]][max_a[1]] / (a[max_a[0]][max_a[0]] - a[max_a[1]][max_a[1]]))) * ((1 - (1 + (2 * a[max_a[0]][max_a[1]] / (a[max_a[0]][max_a[0]] - a[max_a[1]][max_a[1]])) ** 2) ** -0.5) / 2) ** 0.5
		#print(cos_fi, sin_fi)
		
		#формирование матрицы вращения
		v_temp[max_a[0]][max_a[0]] = cos_fi
		v_temp[max_a[0]][max_a[1]] = -sin_fi
		v_temp[max_a[1]][max_a[0]] = sin_fi
		v_temp[max_a[1]][max_a[1]] = cos_fi
		#print(v_temp)
		
		v_prev = np.zeros((n, n))
		a_temp = np.zeros((n, n))
		for j in range(n):
			for k in range(n):
				v_prev[j][k] = v[j][k] # копирование матрицы v
				a_temp[j][k] = a[j][k] # копирование матрицы a
		
		# формирование итоговой матрицы преобразования V и приближённо диагональной матрицы A (A'=A*Vk)
		v = np.zeros((n, n))
		a = np.zeros((n, n))
		for j in range(n):
			for k in range(n):
				for l in range(n):
					v[j][k] += v_prev[j][l] * v_temp[l][k]
					a[j][k] += a_temp[j][l] * v_temp[l][k]
		
		# формирование обратной матрицы к матрице вращения
		v_temp[max_a[0]][max_a[1]] = sin_fi
		v_temp[max_a[1]][max_a[0]] = -sin_fi
		
		# умножение обратной матрицы к матрице вращения v_temp на матрицу a: A''=Vk^(-1)*A'
		a_temp = np.zeros((n, n))
		for j in range(n):
			for k in range(n):
				a_temp[j][k] = a[j][k] # копирование матрицы a
		a = np.zeros((n, n))
		for j in range(n):
			for k in range(n):
				for l in range(n):
					a[j][k] += v_temp[j][l] * a_temp[l][k]
		#print(a)
		#print(v)
	
	eigenvalues = np.zeros(n) # массив собственных значений
	
	ind = np.zeros(n, dtype = int) # массив индексов, связывающих i-ое собственное значение с int[i]-тым столбцом итоговой матрицы преобразования V
	for i in range(n):
		eigenvalues[i] = a[i][i] # занесение собственных значений
		ind[i] = i # занесение соответствующего индекса
	
	# сортировка собственных значений по убыванию
	for i in range(n):
		for j in range(1, n - i):
			if (eigenvalues[j - 1] < eigenvalues[j]):
				swap = eigenvalues[j - 1]
				eigenvalues[j - 1] = eigenvalues[j]
				eigenvalues[j] = swap
				swap = ind[j - 1]
				ind[j - 1] = ind[j]
				ind[j] = swap
				
	# вывод собственных значений
	print('Rotation method:\n\tEigenvalues:')
	for i in range(n):
		print('\t\t', i + 1, ': ', eigenvalues[i])
	
	eigenvectors = np.zeros((n, n)) # массив собственных векторов (матрица)
	for i in range(n):
		# занесение собственных векторов
		for j in range(n):
			eigenvectors[j][i] = v[j][ind[i]]
		# выравнивание собственного вектора по последней координате (последняя координата будет равняться 1)
		for j in range(n):
			eigenvectors[j][i] /= eigenvectors[n - 1][i]
	
	# вывод собственных векторов
	print('\tEigenvectors:\n\t', end = "")
	for i in range(n):
		print('\t', '   {0}   '.format(i + 1), end = "")
	print()
	for i in range(n):
		print('\t\t', end = "")
		for j in range(n):
			print('{0:f}'.format(eigenvectors[i][j]), '\t', end = "")
		print()
	print()
	
	return np.insert(eigenvectors, 0, eigenvalues, 0)

# Степенной метод и метод λ-разности
def Power_and_λ_difference_method(A, error):
	n = A[0].size
	#print(A)
	y = np.zeros(n) # произвольный вектор, подвергаемый преобразованию
	y_full = np.zeros((1, n)) # массив преобразованных векторов
	y[0] = 1
	
	i = 1 # индекс следующего вектора, подвергаемого преобразованию
	not_enough = True # булева переменная, которая следит за окончанием преобразования
	prev = 10e+10 # разность текущего и предыдущего собственных значений 
	while(not_enough):
		for j in range(n):
			y_full[i - 1][j] = y[j] # копирование предыдущего вектора
		for j in range(n):
			y[j] = 0
			for k in range(n):
				y[j] += A[j][k] * y_full[i - 1][k] # преобразование предыдущего вектора
		y_full = np.append(y_full, np.array([y]), axis = 0) # добавление преобразованного вектора в массив векторов
		
		# условие выхода из цикла
		if (abs(y[0] / y_full[i - 1][0] - prev) <= error / 10000):
			not_enough = False
		else:
			prev = y[0] / y_full[i - 1][0]
		i += 1
	
	#print(i - 1)
	#print(y[0] / y_full[i - 2][0])
	eigenvalues = np.zeros(2) # массив собственных значений
	eigenvalues[0] = y[0] / y_full[i - 2][0] # формирование первого собственного значения
	
	# нахождение следующего собственноо значения
	eigenvalues[1] = eigenvalues[0] + 1
	temp = eigenvalues[0] # предыдущее собственное значение, которое нуждается в нахождении
	j = i - 1
	while (abs(eigenvalues[1] - temp) > error * 7 and j > 1):
		temp = eigenvalues[1] # занесение предыдущего значения
		eigenvalues[1] = (y_full[j][2] - eigenvalues[0] * y_full[j - 1][2]) / (y_full[j - 1][2] - eigenvalues[0] * y_full[j - 2][2]) # вычисление собственного значение
		j -= 1
	#print(eigenvalues[1], j)
	
	# вывод собственных значений
	print('Power and λ-difference method:\n\tEigenvalues:')
	for k in range(2):
		print('\t\t', k + 1, ': ', eigenvalues[k])
	#print(eigenvalues)
	
	eigenvectors = np.zeros((n, 2)) # массив собственных векторов (матирца)
	# формирование собственных векторов
	for k in range(n):
		eigenvectors[k][0] = y_full[i - 1][k]
		eigenvectors[k][1] = y_full[j - 4][k] - eigenvalues[0] * y_full[j - 5][k]
	# выравнивание собственных векторов по последней координате (последняя координата будет равняться 1)
	for i in range(2):
		for j in range(n):
			eigenvectors[j][i] /= eigenvectors[n - 1][i]
	
	# вывод собственных векторов
	print('\tEigenvectors:\n\t', end = "")
	for i in range(2):
		print('\t', '   {0}   '.format(i + 1), end = "")
	print()
	for i in range(n):
		print('\t\t', end = "")
		for j in range(2):
			print('{0:f}'.format(eigenvectors[i][j]), '\t', end = "")
		print()
	print()
	#print(eigenvectors)
	
	return np.insert(eigenvectors, 0, eigenvalues, 0)

def main():
	error = 0.5e-4 # требуемая погрешность
	A = np.array([[3.29, -0.75, 0.31, -0.49], [-0.75, 5.24, 0.98, 0.12], [0.31, 0.98, 4.71, 0.32], [-0.49, 0.12, 0.32, 8.92]]) # исходная матрица
	
	# вызов методов
	result = Danilevski_regular_method(A, error)
	result = Danilevski_nonregular_method(A, error)
	result = Leverie_method(A, error)
	result = Fadeev_method(A, error)
	result = Krilov_method(A, error)
	result = Rotation_method(A, error)
	result = Power_and_λ_difference_method(A, error)
	

if __name__ == '__main__':
    main()
