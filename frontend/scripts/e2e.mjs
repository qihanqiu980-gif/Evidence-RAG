import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { spawn } from 'node:child_process'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import puppeteer from 'puppeteer-core'

const externalBaseUrl = process.env.RAG_APP_E2E_URL
const baseUrl = externalBaseUrl ?? 'http://127.0.0.1:8011'
const chromePath =
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
const kbName = `前端自动化验收 ${Date.now()}`
const here = path.dirname(fileURLToPath(import.meta.url))
const projectRoot = path.resolve(here, '../..')

let server
let dataDir
let browser
let invalidUploadPath

const waitForService = async (url) => {
  const deadline = Date.now() + 15000
  for (;;) {
    try {
      const response = await fetch(new URL('/api/status', url))
      if (response.ok) return
    } catch {
      // Retry until the local E2E service is ready.
    }
    if (Date.now() >= deadline) throw new Error(`E2E service did not start: ${url}`)
    await new Promise((resolve) => setTimeout(resolve, 250))
  }
}

const stopServer = async () => {
  if (!server || server.exitCode !== null) return
  server.kill('SIGTERM')
  await Promise.race([
    new Promise((resolve) => server.once('exit', resolve)),
    new Promise((resolve) => setTimeout(resolve, 2000)),
  ])
  if (server.exitCode === null) server.kill('SIGKILL')
}

try {
  if (!externalBaseUrl) {
    dataDir = await mkdtemp(path.join(tmpdir(), 'rag-app-e2e-'))
    server = spawn(
      path.resolve(projectRoot, '.venv/bin/python'),
      [
        // Keep the argument absolute so it is independent of spawn cwd.
        path.resolve(here, 'e2e_server.py'),
        '--port',
        new URL(baseUrl).port,
        '--data-dir',
        dataDir,
      ],
      { cwd: here, stdio: ['ignore', 'pipe', 'pipe'] },
    )
    server.stdout.on('data', (chunk) => process.stdout.write(`[server] ${chunk}`))
    server.stderr.on('data', (chunk) => process.stderr.write(`[server] ${chunk}`))
    server.on('error', (error) => console.error(`[server:error] ${error.message}`))
    await waitForService(baseUrl)
  }

  browser = await puppeteer.launch({
    executablePath: chromePath,
    headless: true,
    args: [
      '--disable-gpu',
      '--no-first-run',
      '--disable-background-networking',
      '--disable-component-update',
    ],
  })

  const page = await browser.newPage()
  let asyncUploadCount = 0
  page.on('console', (message) => console.error(`[console:${message.type()}] ${message.text()}`))
  page.on('pageerror', (error) => console.error(`[pageerror] ${error.message}`))
  page.on('request', (request) => {
    if (request.url().endsWith('/documents/async') && request.method() === 'POST') {
      asyncUploadCount += 1
    }
  })
  await page.setViewport({ width: 1440, height: 900 })
  await page.goto(baseUrl, { waitUntil: 'networkidle0' })
  await page.waitForFunction(() => document.body.innerText.includes('已连接'))

  const clickText = async (text) => {
    await page.evaluate((value) => {
      const element = [...document.querySelectorAll('button')].find((item) =>
        item.textContent?.includes(value),
      )
      if (!(element instanceof HTMLButtonElement)) throw new Error(`Button not found: ${value}`)
      element.click()
    }, text)
  }

  await clickText('知识库')
  await clickText('新建知识库')
  await page.waitForSelector('#kb-name')
  await page.type('#kb-name', kbName)
  await page.waitForFunction(
    (value) => {
      const input = document.querySelector('#kb-name')
      return input instanceof HTMLInputElement && input.value === value
    },
    { timeout: 2000 },
    kbName,
  )
  const createButton = await page.evaluateHandle(() =>
    [...document.querySelectorAll('button')].find(
      (item) => item.textContent?.trim() === '创建',
    ),
  )
  await createButton.dispose()
  await page.click('#create-knowledge-base')
  try {
    await page.waitForFunction(
      (value) => document.body.innerText.includes(value),
      { timeout: 10000 },
      kbName,
    )
  } catch (error) {
    await page.screenshot({ path: '/tmp/rag-app-e2e-create-failed.png', fullPage: true })
    console.error(await page.evaluate(() => document.body.innerText))
    throw error
  }
  await clickText('导入示例资料')
  await page.waitForFunction(() => document.body.innerText.includes('共 6 项，成功 6'), {
    timeout: 30000,
  })

  const invalidUploadDir = await mkdtemp(path.join(tmpdir(), 'rag-app-e2e-upload-'))
  invalidUploadPath = path.join(invalidUploadDir, 'invalid.pdf')
  await writeFile(invalidUploadPath, 'not a valid pdf')
  const fileInput = await page.$('input[type="file"]')
  if (!fileInput) throw new Error('Upload input was not rendered')
  await fileInput.uploadFile(invalidUploadPath)
  const retryButtonVisible = () =>
    page.evaluate(() =>
      [...document.querySelectorAll('button')].some((item) =>
        item.textContent?.includes('重新提交失败文件'),
      ),
    )
  const initialRetryDeadline = Date.now() + 10000
  while (Date.now() < initialRetryDeadline && !(await retryButtonVisible())) {
    await new Promise((resolve) => setTimeout(resolve, 100))
  }
  if (!(await retryButtonVisible())) throw new Error('Failed-file retry button was not rendered')
  await clickText('重新提交失败文件')
  const retryDeadline = Date.now() + 10000
  while (Date.now() < retryDeadline && asyncUploadCount < 2) {
    await new Promise((resolve) => setTimeout(resolve, 100))
  }
  if (asyncUploadCount < 2) throw new Error('Failed-file retry did not submit a fresh upload job')
  const retryFinished = Date.now() + 10000
  while (Date.now() < retryFinished && !(await retryButtonVisible())) {
    await new Promise((resolve) => setTimeout(resolve, 100))
  }
  if (!(await retryButtonVisible())) throw new Error('Retry upload result did not finish')

  await clickText('问答')
  await page.waitForSelector('textarea[aria-label="输入问题"]')
  await page.type('textarea[aria-label="输入问题"]', '星云智联 AX6000 支持 WiFi 6 吗？')
  await clickText('提问')
  try {
    await page.waitForFunction(() => document.body.innerText.includes('回答完成'), {
      timeout: 20000,
    })
  } catch (error) {
    await page.screenshot({ path: '/tmp/rag-app-e2e-failed.png', fullPage: true })
    console.error(await page.evaluate(() => document.body.innerText))
    throw error
  }
  await clickText('查看证据')
  await page.waitForFunction(() => document.body.innerText.includes('回答证据'))

  const chatLayout = await page.evaluate(() => {
    const panel = document.querySelector('[aria-label="执行过程"]')
    const conversation = document.querySelector('[aria-label="会话"]')
    const input = document.querySelector('textarea[aria-label="输入问题"]')
    return {
      panelWidth: panel?.getBoundingClientRect().width ?? 0,
      conversationWidth: conversation?.getBoundingClientRect().width ?? 0,
      inputWidth: input?.getBoundingClientRect().width ?? 0,
    }
  })
  if (chatLayout.panelWidth <= 0) throw new Error('Execution panel was not rendered')
  if (chatLayout.conversationWidth - chatLayout.inputWidth > 80) {
    throw new Error('Question input does not span the conversation frame')
  }

  const answer = await page.evaluate(() => document.body.innerText)
  await page.screenshot({ path: '/tmp/rag-app-e2e-desktop.png' })
  if (!answer.includes('星云智联 AX6000 支持 WiFi 6 双频。 [1]')) {
    throw new Error('Verified answer was not rendered')
  }

  await page.setViewport({ width: 375, height: 812 })
  await clickText('概览')
  await page.waitForFunction(() => document.body.innerText.includes('存储统计'))
  await page.screenshot({ path: '/tmp/rag-app-e2e-mobile.png', fullPage: true })

  console.log('frontend e2e passed')
} finally {
  if (browser) await browser.close()
  await stopServer()
  if (dataDir) await rm(dataDir, { recursive: true, force: true })
  if (invalidUploadPath) {
    await rm(path.dirname(invalidUploadPath), { recursive: true, force: true })
  }
}
