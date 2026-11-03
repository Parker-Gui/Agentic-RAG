# Definition for a binary tree node.
from typing import Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        def dfs(node: Optional[TreeNode], ans=[]):
            if node is None:
                return

            ans.append(node.val)
            dfs(node.left, ans)
            dfs(node.right, ans)
            return ans

        def bfs(node: Optional[TreeNode], ans=[]):
            if node is None:
                return
            bfs(node.left, ans)
            bfs(node.right, ans)
            ans.append(node.val)
            return ans

        left_ans = []
        left_ans = dfs(root.left, left_ans)
        right_ans = []
        right_ans = bfs(root.right, right_ans)
        right_ans.reverse()
        return left_ans == right_ans


if __name__ == "__main__":
    # Example usage
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(2)
    root.left.left = TreeNode(3)
    root.left.right = TreeNode(4)
    root.right.left = TreeNode(4)
    root.right.right = TreeNode(3)

    solution = Solution()
    print(solution.isSymmetric(root))  # Output: True
