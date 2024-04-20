import streamlit as st
from sklearn.datasets import make_circles, make_moons, make_classification
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import cross_val_score
from sklearn.tree import plot_tree
import numpy as np
import warnings
warnings.filterwarnings('ignore')


st.header("Learn decision tree classification hyperparameter-tuning")
# data related settings
with st.sidebar:
	st.header("Dataset selection")
	data = st.selectbox(
						  "Select dataset!",
						  ("moons", "circles", "random")
						  )
	st.header("Dataset modification")
	figure_ = st.checkbox("Show data plot", value = True)
	decision_regions = st.checkbox("Show decision regions")
	decision_tree = st.checkbox("Show decision tree")
	n_samples = st.slider("Number of samples",
						  min_value=100, max_value=1000, step=50)
	random_state = st.number_input("random state:",
								   min_value = 1,
								   max_value = 100,
								   step = 1)
	if (data == "circles") or (data == "moons"):
		noise = st.slider("Amount of noise",
							  min_value=0.0, max_value=1.0, step=0.05)
	if data == "circles":
		factor = st.slider("Factor by which classes saperated.",
				  		min_value=0.0, max_value=0.9, step=0.05)
	if data == "random":
		class_sep = st.number_input("parameter of distance between classes",
									min_value = 0.1,
									max_value = 10.0,
									step = 0.1)


def plot_decision_boundary(X, y, clf):
    h = .02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap = 'Accent', random_state = 42)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap = 'brg')
    plt.title('Decision regions\n', size = 20)
    # plt.ylabel('Feature 2')
    plt.xticks(())
    plt.yticks(())



def choose_dataset(dataset):
	def circle():
		predictors, target = make_circles(n_samples = n_samples,
							shuffle = True,
							noise = noise,
							factor = factor,
							random_state = random_state)
		return predictors, target

	def moon():
		predictors, target = make_moons(n_samples = n_samples,
						  shuffle = True,
						  noise = noise,
						  random_state = random_state)
		return predictors, target

	def random_classification():
		predictors, target = make_classification(n_samples = n_samples,
												 n_features = 2,
												 n_redundant = 0,
												 n_classes = 2,
												 n_clusters_per_class = 1,
												 class_sep = class_sep,
												 random_state = random_state)
		return predictors, target

	if dataset == "circles":
		X, y = circle()
	elif dataset == "moons":
		X, y = moon()
	elif dataset == "random":
		X, y = random_classification()

	return X, y

X, y = choose_dataset(data)


# hyper-parameter related settings
with st.sidebar:
	st.header("Hyper-parameters")

	criterion = st.selectbox(
		"Criterion",
		("gini", "entropy", "log_loss")
	)

	splitter = st.selectbox(
		"Splitter",
		("best", "random")
	)

	maximum_depth = st.number_input("Maximum depth of the tree (0 is None)",
									min_value=0,
									step=1)
	if maximum_depth == 0:
		max_depth = None
	else:
		max_depth = maximum_depth

	minimum_sample_split = st.number_input("Minimum sample split of the tree",
										   min_value=0.01,
										   step=0.01)
	if minimum_sample_split >= 1.0:
		minimum_sample_split = int(minimum_sample_split)

	minimum_sample_leaf = st.number_input("Minimum sample leaf of the tree",
										  min_value=0.01,
										  step=0.01)
	if minimum_sample_leaf >= 1.0:
		minimum_sample_leaf = int(minimum_sample_leaf)

	min_weight_fraction_leaf = st.number_input("Minimum weight fraction leaf of the tree",
											   min_value=0.01,
											   step=0.01)

	max_feature = st.selectbox(
		"max_features",
		("sqrt", "log2", "none", "int", "float")
	)
	if max_feature == "none":
		max_feature = None
	elif max_feature == "int":
		max_feature = st.number_input("integer value for max_features",
									  min_value=1,
									  max_value=np.array(X).shape[1],
									  step=1)
	elif max_feature == "float":
		max_feature = st.number_input("float value for max_features",
									  min_value=0.01,
									  step=0.01)

	random_state = st.selectbox(
								"random_state",
								("none", "value")
	)
	if random_state == "none":
		random_state = None
	elif random_state == "value":
		random_state = st.number_input("value of random_state",
									   min_value=0,
									   step=1)
	max_leaf_nodes = st.selectbox(
								"max_leaf_nodes",
								("none", "value")
	)
	if max_leaf_nodes == "none":
		max_leaf_nodes = None
	elif max_leaf_nodes == "value":
		max_leaf_nodes = st.number_input("value of random_state",
									   min_value=2,
									   step=1)

	min_impurity_decrease = st.number_input("min_impurity_decrease",
											min_value = 0.0,
											step = 0.01)

	ccp_alpha = st.number_input("ccp_alpha",
								min_value = 0.0,
								step = 0.01)

dtc = DecisionTreeClassifier(
							 criterion = criterion,
							 splitter = splitter,
							 max_depth = max_depth,
							 min_samples_split = minimum_sample_split,
							 min_samples_leaf = minimum_sample_leaf,
							 max_features = max_feature,
							 random_state = random_state,
							 max_leaf_nodes = max_leaf_nodes,
							 min_impurity_decrease = min_impurity_decrease,
  							 ccp_alpha = ccp_alpha,
							 )

dtc.fit(X, y)

X_train, X_test, y_train, y_test = train_test_split(X,y,
													test_size = 0.2,
													shuffle = True,
													random_state = 42)

cross_score = cross_val_score(dtc, X_test, y_test, cv = 10)
st.subheader(f"cross val score is {round(cross_score.mean()*100,2)} %")

if figure_:
	figure, ax = plt.subplots(figsize=(10, 4))
	ax.scatter(X[:,0], X[:,1], c = y, cmap = "brg")
	st.pyplot(fig = figure)

if decision_regions:
	fig, ax = plt.subplots(figsize=(10, 4))
	plot_decision_boundary(X,y,dtc)
	st.pyplot(fig)

if decision_tree:
	fig2, ax2 = plt.subplots(figsize=(12, 30))
	plot_tree(dtc)
	st.pyplot(fig2)