# MuJoCo-LiDAR Project Instructions

## Execution Commands

- ALWAYS use `uv run <command>` to execute Python code
- NEVER use `python` or `python3` directly

## Code Quality Standards

When writing or modifying code, you MUST:

1. **Minimize code**: Write the absolute minimum code needed. No extra features, no "nice-to-haves"
2. **Delete ruthlessly**: Remove unused code immediately. Never comment out code—delete it
3. **Refactor immediately**: If you spot poor design while working, fix it before adding features
4. **Prioritize clarity**: Code must be self-explanatory. Clarity > brevity > flexibility
5. **Optimize for performance**: This is GPU-intensive research code. Slow code wastes expensive compute time

## Design Priorities (in order)

1. Clear, understandable code structure
2. Minimal, focused implementation
3. Good architecture over quick functionality
4. Performance optimization
5. Testability over debuggability

## What NOT to do

- Do NOT add defensive code for impossible scenarios
- Do NOT create abstractions for single-use cases
- Do NOT add features beyond what's explicitly requested
- Do NOT leave commented-out code
- Do NOT add unnecessary error handling or validation in internal code

## Git Commit Guidelines

Use Conventional Commits format:
- `feat:` 新功能
- `fix:` 修复 bug
- `docs:` 文档更新
- `style:` 代码格式化
- `refactor:` 重构
- `test:` 测试相关
- `chore:` 构建/工具配置

**重要规则:**
- 及时 commit，保持提交历史清晰
- **永远不在 main 分支 commit**
- 始终在 feature 分支工作
- PR 合并后删除 feature 分支
