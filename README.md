# Hint-Classifier

## Dependencies

* Python 3
* [SciKit Learn](https://scikit-learn.org/stable/install.html)
* [PyTorch](https://pytorch.org/get-started/locally/)
* `pip3 install pytorch-pretrained-bert`
## Git Workflow

### Starting Work on a New Issue

1. Checkout the master branch using `git checkout master`.
2. Create a new branch off of the `master` branch using `git checkout -b <branch-name>`. Ex: `isabel/read_customers` (some name that indicates what issue it is)
3. Push a remote copy of this branch to GitHub using `git push --set-upstream origin <branch-name>`.


### Creating a Pull Request
This should be done when you have completed and tested your work on an issue branch, and you're ready to merge your changes with the `master` branch.

0. Before creating a pull request, make sure to merge any new changes from `master` into your branch using `git merge master`. Resolve any merge conflicts if any.
1. Go to the [GitHub repository](https://github.com/hollowsunsets/MovieStoreDatabase) and click on the **Pull Requests** tab.
2. Click the **New Pull Request** button.
3. Find the name of your branch in the "Example Comparisons" table and click on it.
4. Click the **Create pull request** button.
![Example of pull request UI 1](docs/pull-request-header.png)
5. In the description of the pull request, write "Closes #<Issue Number>". This will associate the pull request with that issue, so when that pull request is merged, the issue will also be closed. Read more about it [here](https://help.github.com/en/github/managing-your-work-on-github/linking-a-pull-request-to-an-issue).
6. Add someone as a reviewer for the pull request.
![Example of pull request UI 2](docs/pull-request-desc.png)
