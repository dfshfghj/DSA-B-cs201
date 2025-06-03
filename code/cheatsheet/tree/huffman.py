import heapq

def huffman(n, weights):
    if n == 1:
        return weights[0]
    heapq.heapify(weights)
    
    total_cost = 0
    while len(weights) > 1:
        w1 = heapq.heappop(weights)
        w2 = heapq.heappop(weights)
        combined_weight = w1 + w2
        total_cost += combined_weight
        heapq.heappush(weights, combined_weight)
    return total_cost

if __name__ == '__main__':
    print(huffman(5, [1, 2, 3, 4, 5]))