from .packmodel import EisenServingMAR


def create_metadata(
        input_name_list,
        input_type_list,
        input_shape_list,
        output_name_list,
        output_type_list,
        output_shape_list,
        model_input_list=None,
        model_output_list=None,
        custom_meta_dict=None,
):
    """
    Facilitates creation of a metadata dictionary for model packaging (MAR). The method renders user-supplied
    information compliant with standard expected format for the metadata.

    It has to be noted that the format of metadata is completely up the user. The only reqirement is that metadata
    should be always supplied as a json-serializable dictionary.

    This method makes metadata more standard by capturing information about model inputs and outputs in fields
    that are conventionally used and accepted across Eisen ecosystem. That is, this method implements a convention
    about the format of metadata

    :param input_name_list: A list of strings representing model input names Eg. ['input'] for single-input model
    :type input_name_list: list
    :param input_type_list: A list of strings for input types Eg. ['ndarray'] matching exp. type for 'input'
    :type input_type_list: list
    :param input_shape_list: A list of shapes (list) representing expected input shape Eg. [[-1, 3, 244, 244]]
    :type input_shape_list: list
    :param output_name_list: List of strings representing model output names Eg. ['logits', 'prediction']
    :type output_name_list: list
    :param output_type_list: List of strings representing model output types Eg. ['ndarray', 'str']
    :type output_type_list: list
    :param output_shape_list: List of shapes (list) for output shape Eg. [[-1, 10], [-1]]
    :type output_shape_list: list
    :param model_input_list: List of input names that should be used as model inputs (default all input_name_list)
    :type model_input_list: list
    :param model_output_list: List of output names that should be obtained from the model (default all output_name_list)
    :type model_output_list: list
    :param custom_meta_dict: A json-serializable dictionary containing custom information (Eg. options or notes)
    :type custom_meta_dict: dict

    :return: Dictionary containing metadata in standardized format

    """
    if model_input_list is None:
        model_input_list = input_name_list

    if model_output_list is None:
        model_output_list = output_name_list

    metadata = {
        'inputs': [],
        'outputs': [],
        'model_input_list': model_input_list,
        'model_output_list': model_output_list,
        'custom': {}
    }

    if custom_meta_dict is None:
        custom_meta_dict = {}

    assert len(input_name_list) == len(input_type_list) == len(input_shape_list)
    assert len(output_name_list) == len(output_type_list) == len(output_shape_list)

    for name, typ, shape in zip(input_name_list, input_type_list, input_shape_list):
        metadata['inputs'].append({
            'name': name,
            'type': typ,
            'shape': shape
        })

    for name, typ, shape in zip(output_name_list, output_type_list, output_shape_list):
        metadata['outputs'].append({
            'name': name,
            'type': typ,
            'shape': shape
        })

    metadata['custom'] = custom_meta_dict

    return metadata
