def filter_results_by_distance(result_dicts):
    # flatten the list of dictionaries
    result_dicts = [item for sublist in result_dicts for item in sublist]
    # Dictionary to store the minimum distance and corresponding document for each unique id
    min_distance_dict = {}
    for result in result_dicts:
        result_id = result['id']
        result_distance = round(result['distance'], 5)
        result_document = result['document']

        if result_id in min_distance_dict:
            # If the id is already in the dictionary, compare distances
            if result_distance < min_distance_dict[result_id]['distance']:
                min_distance_dict[result_id] = {'distance': result_distance, 'document': result_document}
        else:
            # If the id is not in the dictionary, add it
            min_distance_dict[result_id] = {'distance': result_distance, 'document': result_document}

    # Extract the results from the dictionary
    filtered_results = [{'id': key, 'distance': value['distance'], 'document': value['document']} for key, value in min_distance_dict.items()]

    # sort by distance, then by id if distances are equal
    filtered_results = sorted(filtered_results, key=lambda x: (x['distance'], int(x['id'])))

    return filtered_results


