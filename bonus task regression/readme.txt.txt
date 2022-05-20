Context
A series of machining experiments were run on 2" x 2" x 1.5" wax blocks in a CNC milling machine in the System-level Manufacturing and Automation Research Testbed (SMART) at the University of Michigan. Machining data was collected from a CNC machine for variations of tool condition, feed rate, and clamping pressure. Each experiment produced a finished wax part with an "S" shape - S for smart manufacturing - carved into the top face, as shown in test_artifact.jpg (included in the dataset).

Content
General data includes the experiment number, material (wax), feed rate, and clamp pressure. Outputs per experiment include tool condition (unworn and worn tools) and whether or not the tool passed visual inspection.

Time series data was collected from the 18 experiments with a sampling rate of 100 ms and are separately reported in files experiment_01.csv to experiment_18.csv. Each file has measurements from the 4 motors in the CNC (X, Y, Z axes and spindle). These CNC measurements can be used in two ways:

(1) Taking every CNC measurement as an independent observation where the operation being performed is given in the Machining_Process column. Active machining operations are labeled as "Layer 1 Up", "Layer 1 Down", "Layer 2 Up", "Layer 2 Down", "Layer 3 Up", and "Layer 3 Down".

Acknowledgements
This data was extracted using the Rockwell Cloud Collector Agent Elastic software from a CNC milling machine in the System-level Manufacturing and Automation Research Testbed (SMART) at the University of Michigan.

Inspiration
The dataset can be used in classification studies such as:

1. Tool wear detection --- Supervised regression could be performed for identification of worn and unworn cutting tools. Eight experiments were run with an unworn tool while ten were run with a worn tool (see tool_condition column for indication).

2. "task type": "regression",
3. Attribute infformation:
	{
        "run":numerical,
        "DOC":numerical,
        "time":numerical,
        "material":categorical,
        "feed":numerical,
        "case":numerical,
        "smcAC":timeseries,
        "smcDC":timeseries,
        "vib_table":timeseries,
        "vib_spindle":timeseries,
        "AE_table":timeseries,
        "AE_spindle":timeseries,
	"VB":numerical
	}
4. target: VB
