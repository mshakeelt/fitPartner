

class Signals():
    def __init__(self, file_to_write):
        self.output_file = file_to_write

    def __call__(self, comparison_results, angle_diffs):
        completly_alligned = True
        for joint_pair, result in comparison_results.items():
            if not result and result is not None:
                completly_alligned = False
                print(f"Signal: Follower's {joint_pair[0]}-{joint_pair[1]} joint is not alligned with master!")
                self.output_file.write(f"Signal: Follower's {joint_pair[0]}-{joint_pair[1]} joint is not alligned with master!")
                self.output_file.write("\n")
                if angle_diffs[joint_pair] is not None:
                    if angle_diffs[joint_pair] < 0:
                        print(f"Action: Follower Should rotate {joint_pair[1]} anticlockwise by {round(abs(angle_diffs[joint_pair]), 2)} degrees")
                        self.output_file.write(f"Action: Follower Should rotate {joint_pair[1]} anticlockwise by {round(abs(angle_diffs[joint_pair]), 2)} degrees")
                        self.output_file.write("\n")
                    else:
                        print(f"Action: Follower Should rotate {joint_pair[1]} clockwise by {round(abs(angle_diffs[joint_pair]), 2)} degrees")
                        self.output_file.write(f"Action: Follower Should rotate {joint_pair[1]} clockwise by {round(abs(angle_diffs[joint_pair]), 2)} degrees")
                        self.output_file.write("\n")
        if completly_alligned:
            print("Follower is following the master properly!")
            self.output_file.write("Follower is following the master properly!")
            self.output_file.write("\n")
        self.output_file.write("\n")
        print("\n")