def get_Cnk_and_k(parcial_weights: list[float]) -> tuple[float, int]:
	"""
	Pega o menor peso dos pesos parciais:\n
	\tEquação:\n
	\tCnk=min{Ĉn1, Ĉn2, ..., Ĉnm}
	"""
	min_index=0
	min_weight=parcial_weights[0]
	for (i, weight) in enumerate(parcial_weights):
		if weight < min_weight:
			min_weight = weight
			min_index = i
	return (min_weight, min_index)


def get_Ɛ_and_basis_outgoing_index(base_x, y) -> tuple[float, int]:
	"""
	Calcula o valor de epsilon e o index que sairá da base\n
	\tEquação:\n
	\tƐ=min{X̄1/y1, X̄2/y2, ..., X̄n/yn}, y > 0
	"""
	reason_values=[]
	for i_line in range(len(y[:, 0])):
		reason=base_x[i_line][0]/y[i_line][0]
		reason_values.append(reason)
	min_reason=float('inf')
	i_min_reason=0
	for (i_reason, reason) in enumerate(reason_values):
		if reason >= 0 and reason > min_reason:
			min_reason=reason
			i_min_reason=i_reason
	return (min_reason, i_min_reason)

