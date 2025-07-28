def main(db_name):
    # read first layers and second layers from txt file and combine all pairs
    # the output is a txt file of format
    # - - first_layer_row
    #   - second_layer_row
    with open(f'layers_{db_name}.txt', 'w') as handle3:
        with open(f'first_layers_{db_name}.txt', 'r') as handle:
            first_layers = handle.readlines()
            with open(f'second_layers_{db_name}.txt', 'r') as handle2:
                second_layers = handle2.readlines()
                for first_layer in first_layers:
                    for second_layer in second_layers:
                            handle3.write(f'  - {first_layer.strip()}\n')
                            handle3.write(f'    {second_layer.strip()}\n')
        # remove the last newline character
        handle3.seek(handle3.tell() - 1)
        handle3.truncate()

if __name__ == "__main__":
    main('IMDB')