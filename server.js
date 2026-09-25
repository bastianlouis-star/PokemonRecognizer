const http = require('node:http')
const fs = require('node:fs')
const path = require('node:path')
const crypto = require('node:crypto')
const { execFile } = require('node:child_process')

const PORT = 3000
const ROOT = __dirname

const MIME_TYPES = {
    '.html': 'text/html',
    '.js': 'text/javascript',
    '.css': 'text/css',
    '.json': 'application/json',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.svg': 'image/svg+xml',
}

const VENV_PYTHON = path.join(ROOT, '.venv', 'Scripts', 'python.exe')
const SCRIPT_PATH = path.join(ROOT, 'whoIsThatPokemon.py')
const UPLOADS_DIR = path.join(ROOT, 'uploads')
const MAX_UPLOAD_SIZE = 10 * 1024 * 1024

const EXTENSIONS_PAR_TYPE = {
    'image/png': '.png',
    'image/jpeg': '.jpg',
    'image/webp': '.webp',
    'image/gif': '.gif',
}

function gererIdentification(req, res) {
    const morceaux = []
    let taille = 0
    let tropGros = false

    req.on('data', (chunk) => {
        taille += chunk.length
        if (taille > MAX_UPLOAD_SIZE) {
            tropGros = true
            req.destroy()
        } else {
            morceaux.push(chunk)
        }
    })

    req.on('end', () => {
        if (tropGros) {
            res.writeHead(413, { 'Content-Type': 'application/json' })
            res.end(JSON.stringify({ erreur: 'Image trop volumineuse.' }))
            return
        }

        const donnees = Buffer.concat(morceaux)
        if (donnees.length === 0) {
            res.writeHead(400, { 'Content-Type': 'application/json' })
            res.end(JSON.stringify({ erreur: 'Aucune image reçue.' }))
            return
        }

        fs.mkdirSync(UPLOADS_DIR, { recursive: true })
        const extension = EXTENSIONS_PAR_TYPE[req.headers['content-type']] || '.png'
        const cheminImage = path.join(UPLOADS_DIR, `${crypto.randomUUID()}${extension}`)

        fs.writeFile(cheminImage, donnees, (error) => {
            if (error) {
                res.writeHead(500, { 'Content-Type': 'application/json' })
                res.end(JSON.stringify({ erreur: "Impossible d'enregistrer l'image." }))
                return
            }

            execFile(
                VENV_PYTHON,
                [SCRIPT_PATH, 'predict', cheminImage, '--json'],
                // Each call starts Python + TensorFlow and loads the model (~15s
                // on an idle CPU, ~50s+ while a training run is using it)
                { timeout: 180000, maxBuffer: 5 * 1024 * 1024 },
                (error, stdout, stderr) => {
                    fs.unlink(cheminImage, () => {})

                    if (error) {
                        console.error(stderr || error.message)
                        res.writeHead(500, { 'Content-Type': 'application/json' })
                        res.end(JSON.stringify({ erreur: "Échec de l'identification." }))
                        return
                    }

                    try {
                        const resultats = JSON.parse(stdout.trim().split('\n').pop())
                        res.writeHead(200, { 'Content-Type': 'application/json' })
                        res.end(JSON.stringify(resultats[0]))
                    } catch (parseError) {
                        console.error('Réponse Python invalide :', stdout, stderr)
                        res.writeHead(500, { 'Content-Type': 'application/json' })
                        res.end(JSON.stringify({ erreur: 'Réponse du modèle invalide.' }))
                    }
                }
            )
        })
    })
}

const server = http.createServer((req, res) => {
    if (req.method === 'POST' && req.url === '/api/identifier') {
        gererIdentification(req, res)
        return
    }

    const urlPath = req.url === '/' ? '/pokemon.html' : req.url.split('?')[0]
    const filePath = path.join(ROOT, decodeURIComponent(urlPath))

    fs.readFile(filePath, (error, content) => {
        if (error) {
            res.writeHead(404, { 'Content-Type': 'text/plain' })
            res.end('404 - Fichier introuvable')
            return
        }

        const ext = path.extname(filePath)
        res.writeHead(200, { 'Content-Type': MIME_TYPES[ext] || 'application/octet-stream' })
        res.end(content)
    })
})

server.listen(PORT, () => {
    console.log(`Serveur démarré : http://localhost:${PORT}`)
})
